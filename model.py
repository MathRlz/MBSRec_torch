import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward components"""
    
    def __init__(self, hidden_size, num_heads, dropout_rate):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-8)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-8)
        
    def forward(self, x, mask=None, is_training=True):
        # Self-attention with layer norm
        x_norm = self.norm1(x)
        
        # Create causal mask if needed
        attn_mask = None
        if mask is not None:
            seq_len = x.size(1)
            attn_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        
        # Apply attention
        attn_output, _ = self.attention(
            x_norm, x_norm, x_norm, 
            attn_mask=attn_mask, 
            need_weights=False
        )
        x = x + attn_output
        
        # Feed-forward with layer norm
        x_norm = self.norm2(x)
        ff_output = self.feed_forward(x_norm)
        x = x + ff_output
        
        return x

class ItemContextProcessor(nn.Module):
    """Processes items and their context features"""
    
    def __init__(self, item_vocab_size, hidden_size, context_size=4):
        super().__init__()
        self.item_emb = nn.Embedding(item_vocab_size + 1, hidden_size, padding_idx=0)
        self.context_emb = nn.Linear(context_size, hidden_size)
        self.joint_emb = nn.Linear(hidden_size * 2, hidden_size)
        
        # Initialize weights
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.normal_(self.context_emb.weight, std=0.01)
        nn.init.normal_(self.joint_emb.weight, std=0.01)
        
    def forward(self, item_ids, context_features):
        item_emb = self.item_emb(item_ids)
        context_emb = self.context_emb(context_features)
        joint_emb = torch.cat([item_emb, context_emb], dim=-1)
        return self.joint_emb(joint_emb)

class MBSRec(nn.Module):
    def __init__(self, usernum, itemnum, args):
        super().__init__()
        self.usernum = usernum
        self.itemnum = itemnum
        self.args = args
        
        # Item and context processing
        self.item_processor = ItemContextProcessor(
            item_vocab_size=itemnum,
            hidden_size=args.hidden_units,
            context_size=4  # Assuming context size is 4
        )
        
        # Positional encoding
        self.pos_emb = nn.Embedding(args.maxlen, args.hidden_units)
        nn.init.normal_(self.pos_emb.weight, std=0.01)
        
        # Dropout
        self.dropout = nn.Dropout(args.dropout_rate)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=args.hidden_units,
                num_heads=args.num_heads,
                dropout_rate=args.dropout_rate
            ) for _ in range(args.num_blocks)
        ])
        
        # Final normalization
        self.final_norm = nn.LayerNorm(args.hidden_units, eps=1e-8)
    
    def _get_sequence_representation(self, input_seq, seq_cxt, is_training=True):
        """Generate sequence representation from input sequence and context"""
        mask = (input_seq > 0).float().unsqueeze(-1)
        batch_size = input_seq.size(0)
        
        # Process items and context
        seq = self.item_processor(input_seq, seq_cxt)
        
        # Add positional encoding
        positions = torch.arange(input_seq.size(1), device=input_seq.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_emb(positions)
        seq = seq + pos_emb
        
        # Apply dropout during training
        seq = self.dropout(seq) if is_training else seq
        seq = seq * mask
        
        # Transformer blocks
        for block in self.transformer_blocks:
            seq = block(seq, mask=True if is_training else None, is_training=is_training)
            seq = seq * mask  # Apply mask after each block
        
        # Final normalization
        seq = self.final_norm(seq)
        
        return seq, mask
    
    def forward(self, u, input_seq, pos, neg, seq_cxt, pos_cxt, is_training=True, pos_weight=None, neg_weight=None):
        batch_size = input_seq.size(0)
        
        # Get sequence representation
        seq, _ = self._get_sequence_representation(input_seq, seq_cxt, is_training)
        
        # Reshape for prediction
        seq_emb = seq.reshape(batch_size * self.args.maxlen, self.args.hidden_units)
        
        # Reshape target items
        pos = pos.view(batch_size * self.args.maxlen)
        neg = neg.view(batch_size * self.args.maxlen)
        pos_cxt = pos_cxt.view(batch_size * self.args.maxlen, 4)
        
        # Set default weights if not provided
        if pos_weight is not None:
            pos_weight = pos_weight.view(batch_size * self.args.maxlen)
        else:
            pos_weight = torch.ones_like(pos, dtype=torch.float)
            
        if neg_weight is not None:
            neg_weight = neg_weight.view(batch_size * self.args.maxlen)
        else:
            neg_weight = torch.ones_like(neg, dtype=torch.float)
        
        # Process target items
        pos_emb = self.item_processor(pos, pos_cxt)
        neg_emb = self.item_processor(neg, pos_cxt)
        
        # Calculate logits
        pos_logits = torch.sum(pos_emb * seq_emb, dim=-1)
        neg_logits = torch.sum(neg_emb * seq_emb, dim=-1)
        
        # Calculate loss
        istarget = (pos > 0).float()
        pos_loss = F.logsigmoid(pos_logits) * pos_weight * istarget
        neg_loss = F.logsigmoid(-neg_logits) * neg_weight * istarget
        loss = -torch.sum(pos_loss + neg_loss) / torch.sum(istarget)
            
        # Calculate AUC
        auc = torch.sum(
            ((torch.sign(pos_logits - neg_logits) + 1) / 2) * istarget
        ) / torch.sum(istarget)
        
        return loss, auc, seq_emb
    
    def predict(self, u, input_seq, test_item, seq_cxt, test_item_cxt):
        # Get sequence representation (last state)
        seq, _ = self._get_sequence_representation(input_seq, seq_cxt, is_training=False)
        
        # Use last position for prediction
        seq_emb = seq[:, -1]
        
        # Process test items
        test_item_emb = self.item_processor(test_item, test_item_cxt)
        
        # Calculate test logits
        test_logits = torch.matmul(seq_emb, test_item_emb.transpose(0, 1))
        
        return test_logits
