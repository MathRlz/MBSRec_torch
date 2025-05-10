import torch
import numpy as np
import argparse
import time
import os
import csv
from torch.utils.data import DataLoader
from dataset import MBSRecDataset

from model import MBSRec
from util import load_checkpoint, save_checkpoint

def evaluate_valid(model, dataset, args, device):
    train, valid, test, Beh, Beh_w, usernum, itemnum = dataset
    
    model.eval()
    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    
    # Only use users that exist in both train and valid sets
    if usernum > 10000:
        # Get intersection of keys, limited to first 10000
        potential_users = set(train.keys()) & set(valid.keys())
        users = list(potential_users)[:10000]
    else:
        # Get all users that exist in both train and valid
        users = list(set(train.keys()) & set(valid.keys()))
    
    with torch.no_grad():
        for u in users:
            if len(train[u]) < 1 or len(valid[u]) < 1:
                continue
                
            seq = np.zeros([args.maxlen], dtype=np.int32)
            idx = args.maxlen - 1
            for i in reversed(train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1:
                    break
                    
            seq_cxt = []
            for i in seq:
                # Use args.context_size to create correct size context vector
                seq_cxt.append(Beh.get((u, i), [0] * args.context_size))
            seq_cxt = np.array(seq_cxt)
            
            rated = set(train[u])
            rated.add(0)
            item_idx = [valid[u][0]]
            
            testitemscxt = []
            # Use args.context_size here too
            testitemscxt.append(Beh.get((u, valid[u][0]), [0] * args.context_size))
            
            # Sample negative items
            for _ in range(99):
                t = np.random.randint(1, itemnum + 1)
                while t in rated:
                    t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)
                # Use args.context_size here too
                testitemscxt.append(Beh.get((u, t), [0] * args.context_size))
                
            # Convert to tensors
            u_tensor = torch.tensor(np.array([u]), device=device)
            seq_tensor = torch.tensor(np.array([seq]), device=device)
            item_idx_tensor = torch.tensor(item_idx, device=device)
            seq_cxt_tensor = torch.tensor(np.array([seq_cxt], dtype=np.float32), dtype=torch.float, device=device)
            testitemscxt_tensor = torch.tensor(np.array(testitemscxt, dtype=np.float32), dtype=torch.float, device=device)
            
            # Get predictions
            predictions = -model.predict(u_tensor, seq_tensor, item_idx_tensor, seq_cxt_tensor, testitemscxt_tensor)
            predictions = predictions[0].cpu().numpy()
            
            # Calculate metrics
            rank = predictions.argsort().argsort()[0]
            
            valid_user += 1
            
            if rank < 10:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1
                
    return NDCG / valid_user, HT / valid_user

def evaluate_test(model, dataset, args, device):
    train, valid, test, Beh, Beh_w, usernum, itemnum = dataset

    model.eval()
    NDCG = 0.0
    HT = 0.0
    test_user = 0.0

    # Only evaluate users that have test data
    if usernum > 10000:
        users = list(test.keys())[:10000]
    else:
        users = list(test.keys())

    with torch.no_grad():
        for u in users:
            # Skip users with no history or no test item
            if len(train.get(u, [])) < 1 or len(test[u]) < 1:
                continue

            # Build sequence from user's history (train)
            seq = np.zeros([args.maxlen], dtype=np.int32)
            idx = args.maxlen - 1
            for i in reversed(train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1:
                    break

            seq_cxt = []
            for i in seq:
                seq_cxt.append(Beh.get((u, i), [0] * args.context_size))
            seq_cxt = np.array(seq_cxt)

            # The positive test item
            item_idx = [test[u][0]]
            testitemscxt = [Beh.get((u, test[u][0]), [0] * args.context_size)]

            # Sample 99 negative items not in user's history
            rated = set(train[u])
            rated.add(0)
            for _ in range(99):
                t = np.random.randint(1, itemnum + 1)
                while t in rated or t in item_idx:
                    t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)
                testitemscxt.append(Beh.get((u, t), [0] * args.context_size))

            # Convert to tensors
            u_tensor = torch.tensor(np.array([u]), device=device)
            seq_tensor = torch.tensor(np.array([seq]), device=device)
            item_idx_tensor = torch.tensor(item_idx, device=device)
            seq_cxt_tensor = torch.tensor(np.array([seq_cxt], dtype=np.float32), dtype=torch.float, device=device)
            testitemscxt_tensor = torch.tensor(np.array(testitemscxt, dtype=np.float32), dtype=torch.float, device=device)

            # Predict and evaluate
            predictions = -model.predict(u_tensor, seq_tensor, item_idx_tensor, seq_cxt_tensor, testitemscxt_tensor)
            predictions = predictions[0].cpu().numpy()
            rank = predictions.argsort().argsort()[0]

            test_user += 1
            if rank < 10:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1

    return NDCG / test_user, HT / test_user

def save_metrics(metrics_dict, filename):
    """Save metrics to a CSV file with predefined columns"""
    # Define all possible columns upfront
    fieldnames = ['epoch', 'loss', 'auc', 'ndcg', 'hr', 'time']
    
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics_dict)

def train_model(model, dataset, args, device):
    train, valid, test, Beh, Beh_w, usernum, itemnum = dataset

    train_dataset = MBSRecDataset(train, Beh, Beh_w, usernum, itemnum, args.maxlen, args.context_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=5,         # Tune for your CPU
        pin_memory=True
    )
    num_batch = len(train_loader)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    metrics_file = f"{args.train_dir}/training_metrics.csv"
    T = 0.0
    t0 = time.time()
    start_epoch = 1
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, args.resume)

    for epoch in range(start_epoch, args.num_epochs + 1):
        model.train()
        total_loss = 0
        total_auc = 0

        for batch in train_loader:
            # Each batch is already a tuple of tensors
            batch = [x.to(device, non_blocking=True) for x in batch]
            u, seq, pos, neg, seq_cxt, pos_cxt, pos_weight, neg_weight, recency = batch

            loss, auc, _ = model(u, seq, pos, neg, seq_cxt, pos_cxt, True, pos_weight, neg_weight)
            total_loss += loss.item()
            total_auc += auc.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Calculate average loss and AUC for this epoch
        avg_loss = total_loss / num_batch
        avg_auc = total_auc / num_batch
        
        # Initialize metrics dictionary with always-collected metrics
        metrics = {'epoch': epoch, 'loss': avg_loss, 'auc': avg_auc}
        print(f'loss in epoch... {epoch} is {avg_loss}, auc: {avg_auc:.4f}')
        
        # Save checkpoint if specified
        if args.checkpoint_interval > 0 and epoch % args.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, epoch, avg_loss, args)
        
        # Evaluate metrics every 10 epochs
        if epoch % 10 == 0:
            t1 = time.time() - t0
            T += t1
            print('Evaluating')
            ndcg, hr = evaluate_valid(model, dataset, args, device)
            
            # Add these metrics only when collected (every 10 epochs)
            metrics.update({'ndcg': ndcg, 'hr': hr, 'time': T})
            
            print(f'epoch:{epoch}, time: {T}(s), valid (NDCG@10: {ndcg:.4f}, HR@10: {hr:.4f})')
            t0 = time.time()
        
        # Save metrics every epoch
        if (args.log_metrics):
            save_metrics(metrics, metrics_file)
    
    print("Done")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--dataset_type', required=True, type=str,
                        choices=['movie', 'yelp', 'tianchi', 'taobao', 'anime'])
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.0006, type=float)
    parser.add_argument('--maxlen', default=70, type=int)
    parser.add_argument('--hidden_units', default=70, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--num_epochs', default=1001, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.4, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--projection_size', default=8, type=int)
    parser.add_argument('--checkpoint_interval', default=100, type=int, 
                        help='Save checkpoint every n epochs (0 to disable)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--log_metrics', default=True,
                        help='Enable metrics logging to CSV')
    parser.add_argument('--context_size', default=4, type=int,
                        help='Size of context for each item')
    
    args = parser.parse_args()
    
    # Create train directory if it doesn't exist
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    print(f"Device name : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Arguments: {args}")
    
    # Load data
    print("Loading data...")
    if args.dataset_type == 'movie':
        from util import data_partition_movie
        dataset = data_partition_movie(args.dataset)
    elif args.dataset_type in ['taobao', 'tianchi']:
        from util import data_partition_tmall
        dataset = data_partition_tmall(args.dataset)
    elif args.dataset_type == 'yelp':
        from util import data_partition_yelp
        dataset = data_partition_yelp(args.dataset)
    elif args.dataset_type == 'anime':
        from util import data_partition_anime
        dataset = data_partition_anime(args.dataset)
        args.context_size = 11

    [_, _, _, _, _, usernum, itemnum] = dataset
    print("Data loaded.")
    
    # Create model
    model = MBSRec(usernum, itemnum, args).to(device)
    
    # Train model
    train_model(model, dataset, args, device)
    
    # Save final model
    final_model_path = f"{args.train_dir}/model.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}")

if __name__ == '__main__':
    main()
