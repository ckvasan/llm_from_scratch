import tiktoken
import torch
import torch.nn as nn
import sys
sys.path.extend(['c:\\D_Drive\\gpt_model\\attention\\gpt','c:\\D_Drive\\gpt_model'])
from dataloader_prod import create_dataloader_v1
from gpt import generate_simple_text,GPTModel,config
tokeniser = tiktoken.get_encoding('gpt2')
def text_to_tokens(text,device):
    encoded  = tokeniser.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimesnion at index 0
    return encoded_tensor.to(device=device)

def tokens_to_text(tokens):
    flat= torch.tensor(tokens).squeeze(0) # remove batch dimesnion at index 0
    return tokeniser.decode(flat.tolist())




def load_data(filename):
    with open(filename) as f :
        text = f.read()
    return text

text = load_data('data/the-verdict.txt')

train_ratio =0.9

split_index = int(train_ratio * len(text))


train_text = text[:split_index]
val_text = text[split_index:]



train_loader = create_dataloader_v1(train_text, 2, context_length=config['context_length'],stride = config['context_length'],drop_last=True)

val_loader =  create_dataloader_v1(val_text, 2, context_length=config['context_length'],stride = config['context_length'],drop_last=False)

if len(text) * (train_ratio) < config["context_length"]:
    print("Not enough tokens for the training loader. "
          "Try to lower the `GPT_CONFIG_124M['context_length']` or "
          "increase the `training_ratio`")

if len(text) * (1-train_ratio) < config["context_length"]:
    print("Not enough tokens for the validation loader. "
          "Try to lower the `GPT_CONFIG_124M['context_length']` or "
          "decrease the `training_ratio`")

def calc_batch_loss(input_batch,target_batch,model,device):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_batch= input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)
    loss = nn.functional.cross_entropy(logits.flatten(0,1),target_batch.flatten()) #flattening here
    #print(type(loss))
    return loss


def calc_data_loss (dataloader,model,device,num_batches =None):
    total_loss = 0
    if num_batches is None:
        num_batches = len(dataloader)
    for input_batch,target_batch in dataloader:
        loss = calc_batch_loss(input_batch,target_batch,model,device)
        total_loss += loss.item()
    return (total_loss/num_batches)



def test() :
    text = 'every step move forward'
    tokens =  text_to_tokens(text)
    print(tokens)
    print(tokens_to_text(tokens=tokens))
    model = GPTModel(config=config)
    print(model)
    tokens = generate_simple_text(model,idx=text_to_tokens(text),max_new_tokens=10,context_size=config['context_length'])
    print(tokens_to_text(tokens=tokens))

def calculate_loss():
    model = GPTModel(config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_loss = calc_data_loss(train_loader,model=model,device=device)
    val_loss = calc_data_loss(val_loader,model=model,device=device)
    print("Average Training Loss",train_loss)
    print("Average Validation Loss", val_loss)

def train_model(epochs,model, train_loader,optimizer,device, val_loader):
    text = 'The desultory life of the Riviera'
    idx = text_to_tokens(text,device=device)
    loss_per_epochs =[]
    for epoch in range(epochs):
        val = 0
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_batch_loss(input_batch=input_batch,target_batch=target_batch,model=model,device=device)
            val += loss.item()
            loss.backward()
            optimizer.step()
        loss_per_epochs.append(val/len(train_loader))
        
        tokens = generate_simple_text(model,idx=idx,max_new_tokens=10,context_size=config['context_length'])
        print(f"generated text at epoch {epoch}",tokens_to_text(tokens=tokens))
    print(loss_per_epochs)
        
def main():
    torch.manual_seed(123)
    model = GPTModel(config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr = 0.0004,weight_decay=0.1)
    train_model(10,model=model,train_loader=train_loader,optimizer=optimizer,device=device,val_loader=None)
    
if __name__ =='__main__':
    #test()
    main()
