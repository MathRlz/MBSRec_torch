import numpy as np
import torch
from torch.utils.data import Dataset

class MBSRecDataset(Dataset):
    def __init__(self, user_train, Beh, Beh_w, usernum, itemnum, maxlen, context_size):
        self.user_train = user_train
        self.Beh = Beh
        self.Beh_w = Beh_w
        self.usernum = usernum
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.context_size = context_size
        self.valid_users = [u for u in user_train if len(user_train[u]) > 1]

    def __len__(self):
        return len(self.valid_users)

    def __getitem__(self, idx):
        user = self.valid_users[idx]
        seq = np.zeros([self.maxlen], dtype=np.int64)
        pos = np.zeros([self.maxlen], dtype=np.int64)
        neg = np.zeros([self.maxlen], dtype=np.int64)
        recency = np.zeros([self.maxlen], dtype=np.float32)
        recency_alpha = 0.5
        nxt = self.user_train[user][-1]
        ts = set(self.user_train[user])
        idx_ = self.maxlen - 1

        for i in reversed(self.user_train[user][:-1]):
            seq[idx_] = i
            pos[idx_] = nxt
            recency[idx_] = recency_alpha ** (self.maxlen - idx_)
            if nxt != 0:
                # Sample negative
                t = np.random.randint(1, self.itemnum + 1)
                while t in ts:
                    t = np.random.randint(1, self.itemnum + 1)
                neg[idx_] = t
            nxt = i
            idx_ -= 1
            if idx_ == -1:
                break

        seq_cxt = np.array([self.Beh.get((user, i), [0] * self.context_size) for i in seq], dtype=np.float32)
        pos_cxt = np.array([self.Beh.get((user, i), [0] * self.context_size) for i in pos], dtype=np.float32)
        pos_weight = np.array([self.Beh_w.get((user, i), 1.0) for i in pos], dtype=np.float32)
        neg_weight = np.ones_like(pos_weight, dtype=np.float32)

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
            torch.tensor(seq_cxt, dtype=torch.float),
            torch.tensor(pos_cxt, dtype=torch.float),
            torch.tensor(pos_weight, dtype=torch.float),
            torch.tensor(neg_weight, dtype=torch.float),
            torch.tensor(recency, dtype=torch.float),
        )