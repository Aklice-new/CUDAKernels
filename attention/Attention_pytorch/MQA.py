import torch
import torch.nn.functional as F
from torch import nn
import math

class ModelArgs:
    dim: int = 1024         # embedding dimension
    n_head:int = 32         # number of heads



class MultiQureyAttentoin(nn.Module):

    def __init__(self, args : ModelArgs) -> None:
        super().__init__()
        self.head_num = args.n_head
        self.head_dim = args.dim // args.n_head
        self.wq = nn.Linear(args.dim, args.dim, bias=False) # [dim, dim]
        self.wk = nn.Linear(args.dim, self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.head_dim, bias=False)
    
    def forward(self, x : torch.Tensor):
        bs, seq_len, _ = x.shape # [bs, seq_len, dim]
        
        xq = self.wq(x) # [bs, seq_len, dim]
        xk = self.wk(x) # [bs, seq_len, head_dim]
        xv = self.wv(x) # [bs, seq_len, head_dim]

        xq = xq.view(bs, seq_len, self.head_num, self.head_dim)   # [bs, seq_len, head_num, head_dim]
        xk = xk.view(bs, seq_len, 1, self.head_dim)               # [bs, seq_len, 1, head_dim]
        xv = xv.view(bs, seq_len, 1, self.head_dim)               # [bs, seq_len, 1, head_dim]

        xq = xq.transpose(1, 2)        # [bs, head_num, seq_len, head_dim]
        xk = xk.permute(0, 2, 3, 1)    # [bs, 1, head_dim, seq_len]
        xv = xv.transpose(1, 2)        # [bs, 1, seq_len, head_dim]

        score = torch.matmul(xq, xk) / math.sqrt(self.head_dim) # [bs, head_num, seq_len, seq_len]
        score = F.softmax(score, dim=-1)    # [bs, head_num, seq_len, seq_len]

        attention = torch.matmul(score, xv) # [bs, head_num, seq_len, head_dim]
        attention = attention.transpose(1, 2).contiguous().view(bs, seq_len, -1) # [bs, seq_len, dim]
        
        return attention



if __name__ == '__main__':

    mqaAttention = MultiQureyAttentoin(ModelArgs()).to('cuda:0')

    dummy_input = torch.rand([32, 512, 1024], dtype=torch.float32, device='cuda:0')

    output = mqaAttention(dummy_input)
    print(output.shape)

