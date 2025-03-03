import torch
from torch import nn
import torch.nn.functional as F
import math


class ModelArgs:
    dim: int = 1024         # embedding dimension
    n_head:int = 32         # number of heads


class MultiHeadAttention(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.head_dim = args.dim // args.n_head
        self.head_num = args.n_head
        self.wq = nn.Linear(args.dim, args.dim, bias=False) # [dim, dim]
        self.wk = nn.Linear(args.dim, args.dim, bias=False) # [dim, dim]
        self.wv = nn.Linear(args.dim, args.dim, bias=False) # [dim, dim]

    def forward(self, x : torch.Tensor):  # x : [batch, seq_len, dim]
        bs, seq_len, _ = x.shape

        xq = self.wq(x) # [batch, seq_len, dim]
        xk = self.wk(x) # [batch, seq_len, dim]
        xv = self.wv(x) # [batch, seq_len, dim]

        xq = xq.view(bs, seq_len, self.head_num, self.head_dim) # [batch, seq_len, head_num, head_dim]
        xk = xk.view(bs, seq_len, self.head_num, self.head_dim) # [batch, seq_len, head_num, head_dim]
        xv = xv.view(bs, seq_len, self.head_num, self.head_dim) # [batch, seq_len, head_num, head_dim]
        
        xq = xq.transpose(1, 2)     # [batch, head_num, seq_len, head_dim]
        xk = xk.permute(0, 2, 3, 1) # [batch, head_num, head_dim, seq_len] 
        xv = xv.transpose(1, 2)     # [batch, head_num, seq_len, head_dim] 

        score = torch.matmul(xq, xk) / math.sqrt(self.head_dim)   # [batch, head_num, seq_len, seq_len]

        score = F.softmax(score, dim=-1) # [batch, head_num, seq_len, seq_len]

        attention = torch.matmul(score, xv) # [batch, head_num, seq_len, head_dim]
        
        attention = attention.transpose(1, 2).contiguous().view(bs, seq_len, -1) # [batch, seq_len, dim]
    
        return attention
    



if __name__ == '__main__':

    mhaAttention = MultiHeadAttention(ModelArgs()).to('cuda:0')

    dummy_input = torch.rand([32, 512, 1024], dtype=torch.float32, device='cuda:0')

    output = mhaAttention(dummy_input)
    print(output.shape)

