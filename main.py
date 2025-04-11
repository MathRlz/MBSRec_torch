import torch
import numpy as np
import argparse
import time
import os

from model import MBSRec
from util import load_checkpoint, save_checkpoint

def evaluate_valid(model, dataset, args, device):
    train, valid, test, Beh, Beh_w, usernum, itemnum = dataset
    
    model.eval()
    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    
    if usernum > 10000:
        users = list(range(1, 10001))  # Sample users like in original
    else:
        users = range(1, usernum + 1)
    
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
                seq_cxt.append(Beh.get((u, i), [0, 0, 0, 0]))
            seq_cxt = np.array(seq_cxt)
            
            rated = set(train[u])
            rated.add(0)
            item_idx = [valid[u][0]]
            
            testitemscxt = []
            testitemscxt.append(Beh.get((u, valid[u][0]), [0, 0, 0, 0]))
            
            # Sample negative items
            for _ in range(99):
                t = np.random.randint(1, itemnum + 1)
                while t in rated:
                    t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)
                testitemscxt.append(Beh.get((u, valid[u][0]), [0, 0, 0, 0]))
                
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

def train_model(model, dataset, args, device):
    from sampler import WarpSampler
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    # Create data sampler
    train, valid, test, Beh, Beh_w, usernum, itemnum = dataset
    num_batch = len(train) // args.batch_size
    
    # Fix: Pass Beh and Beh_w to the WarpSampler constructor
    sampler = WarpSampler(train, Beh, Beh_w, usernum, itemnum, 
                          batch_size=args.batch_size, 
                          maxlen=args.maxlen, 
                          n_workers=3)
    
    T = 0.0
    t0 = time.time()
    
    # Set starting epoch (for resuming training)
    start_epoch = 1
    
    # Load checkpoint if resuming training
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, args.resume)
    
    for epoch in range(start_epoch, args.num_epochs + 1):
        model.train()
        total_loss = 0
        
        for step in range(num_batch):
            u, seq, pos, neg, seq_cxt, pos_cxt, pos_weight, neg_weight, recency = sampler.next_batch()
            
            # Convert to tensors
            u = torch.tensor(u, device=device)
            seq = torch.tensor(np.array(seq), device=device)
            pos = torch.tensor(np.array(pos), device=device)
            neg = torch.tensor(np.array(neg), device=device)
            seq_cxt = torch.tensor(np.array(seq_cxt, dtype=np.float32), dtype=torch.float, device=device)
            pos_cxt = torch.tensor(np.array(pos_cxt, dtype=np.float32), dtype=torch.float, device=device)
            pos_weight = torch.tensor(np.array(pos_weight, dtype=np.float32), dtype=torch.float, device=device)
            neg_weight = torch.tensor(np.array(neg_weight, dtype=np.float32), dtype=torch.float, device=device)
            recency = torch.tensor(np.array(recency, dtype=np.float32), dtype=torch.float, device=device)
            
            # Forward pass
            auc, loss, _, _ = model(u, seq, seq_cxt, pos, neg, pos_cxt, pos_weight, neg_weight, recency)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batch
        print(f'loss in epoch... {epoch} is {avg_loss}')
        
        # Save checkpoint if specified
        if args.checkpoint_interval > 0 and epoch % args.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, epoch, avg_loss, args)
        
        if epoch % 10 == 0:
            t1 = time.time() - t0
            T += t1
            print('Evaluating')
            ndcg, hr = evaluate_valid(model, dataset, args, device)
            print(f'epoch:{epoch}, time: {T}(s), valid (NDCG@10: {ndcg:.4f}, HR@10: {hr:.4f})')
            t0 = time.time()
    
    sampler.close()
    print("Done")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.0006, type=float)
    parser.add_argument('--maxlen', default=70, type=int)
    parser.add_argument('--hidden_units', default=70, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--num_epochs', default=1001, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.4, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--projection_size', default=8, type=int)
    parser.add_argument('--checkpoint_interval', default=20, type=int, 
                        help='Save checkpoint every n epochs (0 to disable)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    
    # Create train directory if it doesn't exist
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    print("Loading data...")
    from util import data_partition_movie
    dataset = data_partition_movie(args.dataset)
    [train, valid, test, Beh, Beh_w, usernum, itemnum] = dataset
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