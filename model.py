import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import argparse
import copy
from tqdm import tqdm

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate, causality=False):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_units = hidden_units
        self.causality = causality
        
        # Use PyTorch's optimized implementation
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_units,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
    
    def forward(self, queries, keys, mask=None):
        # Handle masking for PyTorch's format
        attn_mask = None
        if mask is not None:
            # PyTorch expects key_padding_mask of shape (batch_size, seq_len)
            # where True values are positions to be masked
            key_padding_mask = (mask.squeeze(1) == 0)  # Invert the mask
            
            # For causal masking
            causal_mask = None
            if self.causality:
                seq_len = queries.size(1)
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=queries.device),
                    diagonal=1
                )
            
            # Apply attention with proper masks
            output, _ = self.attention(
                queries, 
                keys, 
                keys, 
                key_padding_mask=key_padding_mask,
                attn_mask=causal_mask
            )
            return output
        
        # No mask case
        output, _ = self.attention(queries, keys, keys)
        return output

class FeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(FeedForward, self).__init__()
        # Use sequential for better organization
        self.net = nn.Sequential(
            nn.Linear(hidden_units, hidden_units * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units * 4, hidden_units),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, x):
        # Inner layer
        return self.net(x) + x

class MBSRec(nn.Module):
    def __init__(self, usernum, itemnum, args):
        super(MBSRec, self).__init__()
        self.usernum = usernum
        self.itemnum = itemnum
        self.args = args
        
        # Embeddings
        self.item_embedding = nn.Embedding(
            itemnum + 1, args.hidden_units, padding_idx=0)
        self.pos_embedding = nn.Embedding(
            args.maxlen, args.hidden_units)
        
        # Context projections
        self.cxt_projection = nn.Linear(4, args.hidden_units)
        self.feat_projection = nn.Linear(args.hidden_units, args.hidden_units)
        
        # Transformer blocks
        self.blocks = nn.ModuleList()
        for i in range(args.num_blocks):
            self.blocks.append(nn.ModuleDict({
                'attention': MultiHeadAttention(
                    args.hidden_units, args.num_heads, args.dropout_rate, causality=True),
                'norm1': LayerNorm(args.hidden_units),
                'ffn': FeedForward(args.hidden_units, args.dropout_rate),
                'norm2': LayerNorm(args.hidden_units),
            }))
        
        # Final normalization
        self.last_norm = LayerNorm(args.hidden_units)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, u, input_seq, seq_cxt, pos=None, neg=None, pos_cxt=None, pos_weight=None, neg_weight=None, recency=None, is_training=True):
        # Create masks once and reuse - more efficient
        mask = (input_seq != 0).unsqueeze(-1).float()
        attention_mask = mask.squeeze(-1)  # Reuse computation
        
        # Embedding lookup with scaling in one operation
        seq = self.item_embedding(input_seq) * math.sqrt(self.args.hidden_units)
        
        # More efficient position embedding generation
        batch_size, seq_len = input_seq.size()
        positions = torch.arange(seq_len, device=input_seq.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        
        # Combine operations: add positional embedding, then project
        seq = self.feat_projection(seq + pos_emb)
        
        # Process context (note: in original code this is computed but not directly used)
        seq_cxt_emb = self.cxt_projection(seq_cxt)
        
        # Apply dropout and masking in one step
        if is_training:
            seq = F.dropout(seq, p=self.args.dropout_rate) * mask
        else:
            seq = seq * mask
        
        # Transformer blocks
        for block in self.blocks:
            attn_output = block['attention'](seq, seq, attention_mask)
            seq = block['norm1'](seq + attn_output)
            seq = seq * mask
            
            ffn_output = block['ffn'](seq)
            seq = block['norm2'](ffn_output)
            seq = seq * mask
        
        # Final normalization
        seq = self.last_norm(seq)
        
        # Training mode
        if pos is not None and neg is not None:
            # Reshape for predictions
            seq_emb = seq.reshape(-1, self.args.hidden_units)
            pos_emb = self.item_embedding(pos)
            neg_emb = self.item_embedding(neg)
            
            # Process context for items if provided
            if pos_cxt is not None:
                pos_cxt_emb = self.cxt_projection(pos_cxt)  # Not directly used in original
            
            # Project item embeddings
            pos_emb = self.feat_projection(pos_emb)
            neg_emb = self.feat_projection(neg_emb)
            
            # Flatten for prediction
            pos_emb = pos_emb.reshape(-1, self.args.hidden_units)
            neg_emb = neg_emb.reshape(-1, self.args.hidden_units)
            
            # Compute logits
            pos_logits = torch.sum(pos_emb * seq_emb, dim=-1)
            neg_logits = torch.sum(neg_emb * seq_emb, dim=-1)
            
            # Target mask
            pos_flat = pos.reshape(-1)
            istarget = (pos_flat != 0).float()
            
            # Get weights
            if pos_weight is not None:
                pos_weight = pos_weight.reshape(-1)
                neg_weight = neg_weight.reshape(-1)
            else:
                pos_weight = torch.ones_like(pos_flat, dtype=torch.float)
                neg_weight = torch.ones_like(pos_flat, dtype=torch.float)
            
            # Compute loss vectorized
            pos_loss = F.logsigmoid(pos_logits) * pos_weight * istarget
            neg_loss = F.logsigmoid(-neg_logits) * neg_weight * istarget  # Note: log(1-sigmoid(x)) = log(sigmoid(-x))
            loss = -torch.sum(pos_loss + neg_loss) / torch.sum(istarget)
            
            # Compute AUC
            auc = torch.sum(
                ((torch.sign(pos_logits - neg_logits) + 1) / 2) * istarget
            ) / torch.sum(istarget)
            
            return auc, loss, pos_logits, neg_logits
        
        return seq
    
    def predict(self, u, seq, item_idx, seq_cxt, test_item_cxt):
        # Get sequence embeddings (reusing forward logic)
        seq_output = self.forward(u, seq, seq_cxt, is_training=False)
        
        # Get last position output for prediction
        seq_output = seq_output[:, -1, :]  # (batch_size, hidden_units)
        
        # Get item embeddings for test items
        test_item_emb = self.item_embedding(item_idx)
        test_item_cxt_emb = self.cxt_projection(test_item_cxt)  # Process context
        
        # Project test item embeddings
        test_item_emb = self.feat_projection(test_item_emb)
        
        # Compute logits
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        
        return logits