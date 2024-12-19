import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import tiktoken

class GPTDataSet(Dataset):

    def __init__(self, text, tokenizer, context_length, stride):
        super().__init__()
        self.input_ids =[]
        self.target_ids =[]
        token_ids = tokenizer.encode(text,allowed_special={'<|endoftext|>'})
        for i in range(0, len(token_ids)-context_length, stride):
            input_chunks = token_ids[i: i+context_length]
            target_chunks = token_ids[i+1 : i+context_length+1]

            self.input_ids.append(torch.tensor(input_chunks))
            self.target_ids.append(torch.tensor(target_chunks))
     
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index],self.target_ids[index]
    
def create_dataloader(text, batch_size =4,context_length=256, stride=4,shuffle=True):
    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDataSet(text,tokenizer,context_length,stride)

    dataloader = DataLoader(dataset,batch_size = batch_size,shuffle=shuffle,drop_last=True) 
    return dataloader

def load_text(filename:str):
    with open(filename) as f :
        contents = f.read()
    return contents

def create_batched_embeddings(vocab_size,context_length,output_dim):
    #vocab_size = 50257
    #output_dim = 256
    #context_length = 1024

    token_embedding_layer = torch.nn.Embedding(vocab_size,output_dim)
    pos_embedding_layer = torch.nn.Embedding(context_length,output_dim)
    max_length =  context_length
    dataloader = create_dataloader(load_text('data/the-verdict.txt'), batch_size=2,context_length=max_length,stride=4)    
    input_embeddings=[]
    pos_embedding = pos_embedding_layer(torch.arange(max_length)) #max_length - context_length is limited to 4
    for batch in dataloader:
        x,_ = batch
        print(x.shape)
        token_embedding = token_embedding_layer(x)
        input_embedding  = token_embedding + pos_embedding
        input_embeddings.append(input_embedding)
        
    all_input_embeddings = torch.stack(input_embeddings)   
   
    return input_embedding,all_input_embeddings
    


class MaskedAttention(nn.Module) :

    def __init__(self,d_in,d_out,context_length, dropout, bias = False) :
        super().__init__()
        self.W_query = nn.Linear(d_in,d_out)
        self.W_key = nn.Linear(d_in,d_out)
        self.W_value = nn.Linear(d_in,d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask',torch.triu(torch.ones(context_length,context_length),diagonal=1)) 


    def forward(self,x ):
        b, n_tokens, d_in = x.shape 
        # x.shape :  torch.Size([8, 4, 256])
        # b-> batch/rows (8)
        # n_tokens -> number of tokens (which is context_length - 4) of each row
        # d_in -> input embedding dimension (256)
        
        keys = self.W_key(x)

        queries = self.W_query(x)
        values = self.W_value(x)

        atten_scores = queries @ keys.transpose(1,2)
        atten_scores.masked_fill_(self.mask.bool()[:n_tokens,:n_tokens],-torch.inf)

        atten_weights = torch.softmax(atten_scores/keys.shape[-1] ** 0.5, dim =-1)

        atten_weights= self.dropout(atten_weights)

        context_vec= atten_weights@values

        return context_vec


class MultiHeadMaskedAttention(nn.Module):

    def __init__(self, d_in,d_out,context_length, dropout, num_heads, bias=False):
        super().__init__()
        d_out = d_out//num_heads ## important step to divide d_out by num_heads
        self.heads = nn.ModuleList([MaskedAttention(d_in=d_in, d_out=d_out,context_length=context_length,dropout=dropout,bias=bias) for _ in range(num_heads)])
        self.out_proj = nn.Linear(d_out*num_heads, d_out*num_heads) 
    
    def forward(self,x):
        context_vec= torch.cat([head(x) for head in self.heads],dim=-1)
        return self.out_proj(context_vec)

if __name__ =="__main__":

    one_batch, all_batches =  create_batched_embeddings()
    print(one_batch)
    mmah = MultiHeadMaskedAttention(256,128,4,0.0,2)
    out_mmah = mmah(one_batch)
    print(out_mmah.shape)
