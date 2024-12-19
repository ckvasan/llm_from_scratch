import tiktoken
import torch
import numpy as np

from torch.utils.data import DataLoader,Dataset


class GPTDataSet(Dataset):

    def __init__(self,txt,tokenizer, context_length, stride):
        super().__init__()
        self.input_ids =[]
        self.target_ids= []
        tokens = tokenizer.encode(txt,  allowed_special={"<|endoftext|>"})
        for i in range(0,len(tokens)- context_length,stride):
            input_chunks = tokens[i : i+ context_length]
            target_chunks = tokens[i+1 : i+ context_length+1]
            self.input_ids.append(torch.tensor(input_chunks))
            self.target_ids.append(torch.tensor(target_chunks))

        
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index],self.target_ids[index]


def read_file(filename:str):
    with open(filename) as f:
        raw_text = f.read()

    return raw_text

def create_dataloader_v1(txt, batch_size=4, context_length=256, 
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
   
   tokenizer =  tiktoken.get_encoding("gpt2")
   dataset = GPTDataSet(txt,tokenizer,context_length,stride)

   dataloader =  DataLoader(dataset, batch_size=batch_size, shuffle= shuffle,drop_last=drop_last, num_workers= num_workers)
   
   return dataloader

if __name__ =='__main__':

    vocab_size = 50257
    output_dim = 256
    context_length = 1024

    token_embedding_layer = torch.nn.Embedding(vocab_size,output_dim)
    pos_embedding_layer = torch.nn.Embedding(context_length,output_dim)
    max_length =  4
    dataloader = create_dataloader_v1(read_file('data/the-verdict.txt'), batch_size=8,context_length=max_length,stride=4)    

    print(type(dataloader))

    input_embeddings = []
    pos_embedding = pos_embedding_layer(torch.arange(max_length)) #max_length - context_length is limited to 4
    for batch in dataloader:
        x,y = batch
        token_embedding = token_embedding_layer(x)
        input_embedding  = token_embedding + pos_embedding
        input_embeddings.append(input_embedding)
    all_input_embeddings = torch.stack(input_embeddings)    
    print(input_embedding.shape)
    print(all_input_embeddings.shape)
    