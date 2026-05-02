from dataclasses import dataclass
import torch
import torch.nn as nn
from .embedding import TransformerEmbedding
from .encoder_block import EncoderBlock, LayerNorm

@dataclass
class EncoderConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

class SimpleEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            embed = TransformerEmbedding(config),
            h = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        # 对于 Encoder 任务（如分类），这里可以定义不同的 Head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx):
        # 1. Embedding[cite: 1]
        x = self.transformer.embed(idx)
        
        # 2. 堆叠的 Encoder Blocks[cite: 1]
        for block in self.transformer.h:
            x = block(x)
        
        # 3. Final LayerNorm[cite: 1]
        x = self.transformer.ln_f(x)
        
        # 4. Output logits[cite: 1]
        return self.lm_head(x)
