import torch
import torch.nn as nn
import sys
sys.path.append('c:\\D_Drive\\gpt_model\\attention')
print(sys.path)
import attention_mechanism as asm
import tiktoken

config ={
    'emb_dim' : 768,
    'context_length': 256, #1024
    'vocab_size': 50257,
    'n_heads': 12,
    'n_layers': 12,
    'drop_rate':0.1,
    'bias': False  

}

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
         return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):

    def __init__(self,config):
        super().__init__()
       
        self.layers = nn.Sequential(
            nn.Linear(config['emb_dim'], 4 * config['emb_dim']),
            GELU(),
            nn.Linear(4 * config['emb_dim'],config['emb_dim'])
        )
    
    def forward(self,x):
        return self.layers(x)

class LayerNorm(nn.Module):
    def __init__(self,emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.ones = nn.Parameter(torch.ones(emb_dim))
        self.zeros = nn.Parameter(torch.zeros(emb_dim))

    def forward(self,x):
        mean =  x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim= True, unbiased = False)
        norm_x = (x-mean)/torch.sqrt(var+self.eps)
        return self.ones* norm_x + self.zeros

class Transformer(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.attn_block = asm.MultiHeadMaskedAttention(config['emb_dim'],config['emb_dim'],config['context_length'],config['drop_rate'],config['n_heads'], config['bias'])
        self.ff = FeedForward(config=config)
        self.norm1 = LayerNorm(config['emb_dim'])
        self.norm2 = LayerNorm(config['emb_dim'])
        self.dropout = nn.Dropout(config['drop_rate'])
    
    def forward(self,x):
        # x is a single batch which consist of 2 rows and each row consists of 1024 tokens (context_length) and each token is of 768 embidding
        # x.shape ==> single batch ==> torch.Size([2, 1024, 768])
        # **** example : 
        # the input [x] formed by 
        # token_embedding_layer = nn.Embedding(config['vocab_size'], config['emb_dim'])
        # pos_embedding_layer = nn.Embedding(config['context_length], config['emb_dim])
        # eg tokens =[[27,89,7,8,.....,7,8,129],
        #             [6,78,7,8,.....,71,81,178]] tokens.shape is (2,1024)
        # x = token_embedding_layer(tokens) + token_embedding_layer(tokens)
        # x.shape is (2,1024,768)
        # shortcut for attention block
        shortcut = x
        x= self.norm1(x) # input _normilaztion 
        x = self.attn_block(x)
        x = self.dropout(x)
        x =x+shortcut # shortcut is initial embedding gets added to avoid vanishing gradients during backpropagation

        # shortcut for Feedforward layer 
        shortcut = x
        x= self.norm2(x)
        x= self.ff(x)
        x=self.dropout(x)
        x= x+ shortcut

        return x

class GPTModel(nn.Module):

    def __init__(self,config:dict):
        
        super().__init__()
        self.config = config
        self.token_embedding_layer = nn.Embedding(config['vocab_size'],config['emb_dim'])
        self.pos_embedding_layer = nn.Embedding(config['context_length'], config['emb_dim'])
        self.drop_emb = nn.Dropout(config["drop_rate"])
        self.transformer_blocks =nn.Sequential(
            *[Transformer(config=config) for _ in range(config['n_layers'])]
        )
        self.final_norm = LayerNorm(config['emb_dim'])
        self.out_head = nn.Linear(config['emb_dim'],config['vocab_size'],bias=False)
    
    def forward(self,x):
        # x is single batch --> 
        #           [[1,2,3...,3,4,12],
        #           [12,23,34...,43,64,122]]
        # x.shape --> torch.Size([2,1024]) 
        # where a bactch consists of 2 rows and 1024 words of context_length 
        
        ##===================================================================##########
        b,t = x.shape
        
        token_embedding = self.token_embedding_layer(x) 
        pos_embedding = self.pos_embedding_layer(torch.arange(t,device=x.device))

        x = token_embedding + pos_embedding
        
        #x.shape --> torch.Size([2,1024,768]) 
        ##======================================================================##########
        x = self.drop_emb(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

def generate_simple_text(model,idx,max_new_tokens,context_size):
    for _ in range(max_new_tokens):
        idx = idx[:,-context_size:]
        with torch.no_grad():
            logits=  model(idx)
        
        logits = logits[:,-1,:]
        #print(">>",logits) 
        idx_next = torch.argmax(logits,dim=-1,keepdim=True)
        ##print("idx_next",idx_next)
        idx = torch.cat((idx,idx_next),dim=1)
    return idx

if __name__=='__main__':
        #create_batched_embeddings(vocab_size,context_length,output_dim)

        """ 
        torch.manual_seed(123)
        model = GPTModel(config=config)
        model.eval()  # disable dropout

        start_context = "Hello, I am"

        tokenizer = tiktoken.get_encoding("gpt2")
        encoded = tokenizer.encode(start_context)
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)
        print(encoded_tensor,"------------", encoded_tensor.shape)

 """        
        tokenizer = tiktoken.get_encoding("gpt2")
        batched_dataset =asm.create_dataloader(asm.load_text('data/the-verdict.txt'), batch_size =2,context_length=config['context_length'], stride=4,shuffle=True)
        for batch in batched_dataset:
            x,y = batch
            break
        
        model = GPTModel(config=config)
        print(x)
        x= x[-1,:] # Taking last row of a batch
        print(x.shape)
        x= x.unsqueeze(0) # adding a dummy dimension at index 0
        print(x.shape)
        logits = model(x)
        logits = logits[:,-1,:]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)
        print(idx_next)
        print(logits.shape)
        out = generate_simple_text(
        model=model,
        idx=x,
        max_new_tokens=10,
        context_size=config["context_length"]
        )
        decoded_text = tokenizer.decode(out.squeeze(0).tolist())

        print(decoded_text)
