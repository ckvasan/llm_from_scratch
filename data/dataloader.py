from importlib.metadata import version

print("torch version:", version("torch"))
print("tiktoken version:", version("tiktoken"))

import tiktoken
import torch
from torch.utils.data import DataLoader,Dataset

class GPTDataSet(Dataset):
    def __init__(self, tokeniser, text, context_length, stride):
        self.input_ids =[]
        self.target_ids =[]

        # tokenising the input text
        token_ids=tokeniser.encode(text, allowed_special={"<|endoftext|>"})

        for i in range(0,len(token_ids)-context_length, stride):
            input_chunk = token_ids[i:context_length]
            target_chunk = token_ids[i+1:context_length +1]
            self.input_ids = self.input_ids.append(input_chunk)
            self.target_ids = self.target_ids.append(target_chunk)
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index],self.target_ids[index]