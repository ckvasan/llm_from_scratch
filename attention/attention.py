import torch
import torch.nn as nn
import tiktoken
from attention_mechanism import create_batched_embeddings

class MaskedAttention(nn.Module):

    def __init__(self,d_in,d_out,context_length,dropout):
        super().__init__()
        self.W_query = nn.Linear(d_in,d_out)
        self.W_key = nn.Linear(d_in,d_out)
        self.W_value= nn.Linear(d_in,d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask",torch.triu(torch.ones(context_length,context_length),diagonal=1))

    def forward(self,x):
        # x.shape :  torch.Size([8, 4, 256])
        # b -> single batch(8 rows)
        # n_tokens -> number of tokens in each context (4)
        # emb_dim -> each token of embedding_dimesion of 256 
        b,n_tokens,_ = x.shape
        query = self.W_query(x)
        key = self.W_key(x)
        value = self.W_value(x)

        atten_scores = query @ key.transpose(1,2)
        atten_scores.masked_fill_(self.mask.bool()[:n_tokens,:n_tokens],-torch.inf)
        atten_weights = torch.softmax(atten_scores/key.shape[-1] ** 0.5, dim =-1)
        atten_weights= self.dropout(atten_weights)

        context_vec= atten_weights@value
        return context_vec
    
class MaskedMultiHeadAttention(nn.Module):

        def __init__(self,d_in,d_out,context_length,dropout,num_heads):
            super().__init__()
            self.heads= nn.ModuleList([MaskedAttention(d_in,d_out,context_length,dropout) for _ in range(num_heads)])
            self.out_proj = nn.Linear(d_out*num_heads,d_out * num_heads)
        
        def forward(self,x):
            context_vec = torch.cat([head(x) for head in self.heads],dim=-1)
            return self.out_proj(context_vec)    
    
if __name__ =="__main__":

    one_batch, all_batches =  create_batched_embeddings()
    print(one_batch)
    mmah = MaskedMultiHeadAttention(256,128,4,0.0,2)
    out_mmah = mmah(one_batch)
    print(out_mmah.shape)