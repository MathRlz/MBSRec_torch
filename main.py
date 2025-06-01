import torch
import numpy as np
import argparse
import time
import os
import csv
import logging
from torch.utils.data import DataLoader
from dataset import MBSRecDataset, EvalDataset
from tqdm import tqdm

from model import MBSRec
from util import load_checkpoint, save_checkpoint

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--dataset_type', required=True, type=str,
                        choices=['movie', 'yelp', 'tianchi', 'taobao', 'anime'])
    parser.add_argument('--patience', default=10, type=int,
                        help='Number of epochs with no improvement after which training will be stopped')
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
    parser.add_argument('--log_to_file', action='store_true', help='Enable logging to log.txt in train_dir')
    return parser.parse_args()

def setup_logging(log_to_file, train_dir):
    handlers = [logging.StreamHandler()]
    if log_to_file:
        log_path = os.path.join(train_dir, 'log.txt')
        handlers.append(logging.FileHandler(log_path, mode='w'))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=handlers
    )

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"Device name : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    return device

def load_data(args):
    logging.info("Loading data...")
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
    logging.info("Data loaded.")
    return dataset

def save_metrics(metrics_dict, filename):
    fieldnames = ['epoch', 'loss', 'auc', 'ndcg', 'hr', 'time']
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics_dict)

def evaluate_valid_loader(model, valid_loader, device):
    model.eval()
    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    with torch.no_grad():
        for batch in valid_loader:
            u, seq, item_idx, seq_cxt, testitemscxt = [x.to(device) for x in batch]
            predictions = -model.predict(u, seq, item_idx, seq_cxt, testitemscxt)  # [batch_size, num_candidates]
            for i in range(predictions.size(0)):
                pred_row = predictions[i]  # This is a 1D tensor on device
                # Move only this row to CPU for numpy ops
                rank = pred_row.cpu().numpy().argsort().argsort()[0]
                valid_user += 1
                if rank < 10:
                    NDCG += 1 / np.log2(rank + 2)
                    HT += 1
    return NDCG / valid_user, HT / valid_user

def evaluate_test_loader(model, test_loader, device):
    model.eval()
    NDCG = 0.0
    HT = 0.0
    test_user = 0.0
    with torch.no_grad():
        for batch in test_loader:
            u, seq, item_idx, seq_cxt, testitemscxt = [x.to(device) for x in batch]
            predictions = -model.predict(u, seq, item_idx, seq_cxt, testitemscxt)
            predictions = predictions.cpu().numpy()
            for i in range(predictions.shape[0]):
                rank = predictions[i].argsort().argsort()[0]
                test_user += 1
                if rank < 10:
                    NDCG += 1 / np.log2(rank + 2)
                    HT += 1
    return NDCG / test_user, HT / test_user

def train_model(model, dataset, args, device):
    train, valid, test, Beh, Beh_w, usernum, itemnum = dataset

    train_dataset = MBSRecDataset(train, Beh, Beh_w, usernum, itemnum, args.maxlen, args.context_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=5,
        pin_memory=True
    )
    num_batch = len(train_loader)

    valid_dataset = EvalDataset(valid, train, Beh, itemnum, args.maxlen, args.context_size)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    metrics_file = f"{args.train_dir}/training_metrics.csv"
    T = 0.0
    t0 = time.time()
    start_epoch = 1
    best_ndcg = -float('inf')
    epochs_no_improve = 0
    if args.resume:
        start_epoch, best_ndcg, epochs_no_improve = load_checkpoint(model, optimizer, scheduler, args.resume)

    for epoch in tqdm(range(start_epoch, args.num_epochs + 1), desc="Training Epochs", unit="epoch"):
        model.train()
        total_loss = 0
        total_auc = 0

        for batch in train_loader:
            batch = [x.to(device, non_blocking=True) for x in batch]
            u, seq, pos, neg, seq_cxt, pos_cxt, pos_weight, neg_weight, recency = batch

            loss, auc, _ = model(u, seq, pos, neg, seq_cxt, pos_cxt, True, pos_weight, neg_weight)
            total_loss += loss.item()
            total_auc += auc.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / num_batch
        avg_auc = total_auc / num_batch

        t1 = time.time() - t0
        T += t1

        metrics = {'epoch': epoch, 'loss': avg_loss, 'auc': avg_auc, 'time': T}

        if args.checkpoint_interval > 0 and epoch % args.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, avg_loss, args, best_ndcg, epochs_no_improve)

        # Early stopping check every 10 epochs
        if epoch % 10 == 0:
            logging.info('Evaluating')
            ndcg, hr = evaluate_valid_loader(model, valid_loader, device)
            metrics.update({'ndcg': ndcg, 'hr': hr})
            logging.info(f'epoch:{epoch}, loss: {avg_loss}, auc: {avg_auc:.4f}, time: {T}(s), valid (NDCG@10: {ndcg:.4f}, HR@10: {hr:.4f})')

            # Early stopping logic
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                epochs_no_improve = 0
                torch.save(model.state_dict(), f"{args.train_dir}/best_model.pt")
            else:
                epochs_no_improve += 1
                logging.info(f"No improvement in NDCG for {epochs_no_improve} evals.")

            if epochs_no_improve >= args.patience:
                logging.info(f"Early stopping triggered at epoch {epoch}. Best NDCG: {best_ndcg:.4f}")
                break

        if args.log_metrics:
            save_metrics(metrics, metrics_file)

        scheduler.step() 

        t0 = time.time()

    logging.info("Done")

def main():
    args = parse_args()
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)
    setup_logging(args.log_to_file, args.train_dir)
    logging.info(f"Arguments: {args}")

    device = get_device()

    dataset = load_data(args)
    [_, _, _, _, _, usernum, itemnum] = dataset
    model = MBSRec(usernum, itemnum, args).to(device)
    train_model(model, dataset, args, device)

    final_model_path = f"{args.train_dir}/model.pt"
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Final model saved at {final_model_path}")

    # Load the best model for testing
    best_model_path = f"{args.train_dir}/best_model.pt"
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        logging.info(f"Loaded best model from {best_model_path} for testing")
    else:
        logging.warning("Best model not found, using final model for testing")

    test_dataset = EvalDataset(dataset[2], dataset[0], dataset[3], itemnum, args.maxlen, args.context_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    ndcg, hr = evaluate_test_loader(model, test_loader, device)
    logging.info(f"Test NDCG@10: {ndcg:.4f}, HR@10: {hr:.4f}")

if __name__ == '__main__':
    main()
