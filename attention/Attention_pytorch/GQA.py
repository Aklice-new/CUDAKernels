import torch
from torch import nn
import torch.nn.functional as F
import math



class ModelArgs:
    dim:int= 1024
    n_head = 32
    n_group = 8


class GroupQueryAttention(nn.Module):

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.dim = args.dim
        self.head_num = args.n_head
        self.group_num = args.n_group
        self.head_dim = args.dim // args.n_head

        self.wq = nn.Linear(self.dim, self.dim, bias=False)
        self.wk = nn.Linear(self.dim, self.group_num * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.group_num * self.head_dim, bias=False)

    def forward(self, x : torch.Tensor):
        bs, seq_len, _ = x.shape # [bs, seq_len, dim]

        xq = self.wq(x).view(bs, seq_len, self.head_num, self.head_dim)  # [bs, seq_len, head_num, head_dim]
        xk = self.wk(x).view(bs, seq_len, self.group_num, self.head_dim) # [bs, seq_len, group_num, head_dim]
        xv = self.wv(x).view(bs, seq_len, self.group_num, self.head_dim) # [bs, seq_len, group_num, head_dim]

        xq = xq.transpose(1, 2)         # [bs, head_num, seq_len, head_dim]
        xk = xk.permute(0, 2, 3, 1)     # [bs, group_num, head_dim, seq_len]
        xv = xv.transpose(1, 2)         # [bs, group_num, seq_len, head_dim]

        # [bs, head_num (from group_num expanded), head_dim, seq_len]
        xk = xk[:, :, None, :, :].expand(bs, self.group_num, self.head_num // self.group_num, self.head_dim, seq_len).reshape(bs, self.head_num, self.head_dim, seq_len)
        # [bs, head_num (from group_num expanded), seq_len, head_dim]
        xv = xv[:, :, None, :, :].expand(bs, self.group_num, self.head_num // self.group_num, seq_len, self.head_dim).reshape(bs, self.head_num, seq_len, self.head_dim)            

        score = torch.matmul(xq, xk)      # [bs, head_num, seq_len, seq_len]
        score = F.softmax(score, dim=-1)  # [bs, head_num, seq_len, seq_len]

        attention = torch.matmul(score, xv) # [bs, head_num, seq_len, head_dim]
        attention = attention.transpose(1, 2).contiguous().view(bs, seq_len, -1)

        return attention


if __name__ == '__main__':

    gqaAttention = GroupQueryAttention(ModelArgs()).to('cuda:0')

    dummy_input = torch.rand([32, 512, 1024], dtype=torch.float32, device='cuda:0')

    output = gqaAttention(dummy_input)
    print(output.shape)








