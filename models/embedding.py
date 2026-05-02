import torch
import torch.nn as nn

class TransformerEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 词向量[cite: 1]
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # 位置向量[cite: 1]
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        tok_emb = self.wte(idx) 
        pos_emb = self.wpe(pos)
        return self.drop(tok_emb + pos_emb)[cite: 1]
