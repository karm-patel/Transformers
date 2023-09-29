import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch
from datasets import load_dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import os
from dataloader.dataloader import NMTDataset
from torch.utils.data import DataLoader
from models.transformer import Transformer
from torchsummary import torchsummary
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchmetrics import ExactMatch, Accuracy

device = torch.device("cuda")
cpu_device = torch.device("cpu")

# %reload_ext autoreload
# %autoreload 2

n_train = 36000
train_dataset = load_dataset('cfilt/iitb-english-hindi', split=f"train[:{n_train}]")
print(len(train_dataset))

test_dataset = load_dataset('cfilt/iitb-english-hindi', split="test")
print(len(test_dataset))

val_dataset = load_dataset('cfilt/iitb-english-hindi', split="validation")
print(len(val_dataset))

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset))
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))

tr = next(iter(train_dataloader))
vl = next(iter(val_dataloader))
tt = next(iter(test_dataloader))

for lang in ["hi", "en"]:
    tokenizer_path = f"dataset/vocab_{lang}_{n_train}.json"
    if os.path.exists(tokenizer_path):
        # load tokenizer
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['<unk>', '<pad>', '<sos>', '<eos>'], min_frequency=2)
        tokenizer.train_from_iterator(tr['translation'][lang] + vl['translation'][lang]+ tt['translation'][lang], trainer=trainer)

        # save tokenizer
        tokenizer.save(tokenizer_path)


tokenizer_src = Tokenizer.from_file(f"dataset/vocab_en_{n_train}.json")
tokenizer_tar = Tokenizer.from_file(f"dataset/vocab_hi_{n_train}.json")
tokenizer_tar

train_ds = NMTDataset(tokenizer_src, tokenizer_tar, train_dataloader)
val_ds = NMTDataset(tokenizer_src, tokenizer_tar, val_dataloader)
print(len(train_ds), len(val_ds))

B = 512
train_dl = DataLoader(train_ds, batch_size=B, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=len(val_ds), shuffle=True)

seq_len = 100
d_model = 512
drop_prob = 0.1
n_h = 8
d_ff = 2048
d_k = d_model/n_h
n_layers = 6

transformer = Transformer(tokenizer_src.get_vocab_size(), tokenizer_tar.get_vocab_size(), d_model, n_h, n_layers , d_ff, seq_len, drop_prob, tokenizer_src.token_to_id("<pad>"), tokenizer_tar.token_to_id("<pad>")).to(device)

# torchsummary.summary(transformer, [(1, 100), (1, 100)], device="cpu")

### Train loop

import torch.nn.functional as F
import torch.nn as nn

lr = 1e-4
optim = torch.optim.Adam(transformer.parameters(),lr=lr)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('<pad>'))
EM = ExactMatch(task="multiclass", num_classes=tokenizer_tar.get_vocab_size(), 
                ignore_index=tokenizer_tar.token_to_id("<pad>")).to(device)
Acc = Accuracy(task="multiclass", num_classes=tokenizer_tar.get_vocab_size(), 
                ignore_index=tokenizer_tar.token_to_id("<pad>")).to(device)

n_epochs = 30

train_losses, val_losses = [], []
val_ems, val_accs = [], []
prev_loss = 1e4
for i in range(n_epochs):
    transformer.train()
    tqdm_obj = tqdm(train_dl)
    for src, tar in tqdm_obj:
        src, tar = src.to(device), tar.to(device)
        optim.zero_grad()
        logits = transformer(src, tar)
        y_pred = torch.argmax(logits, dim=-1)
        loss = loss_fn(logits.permute(0,2,1), tar)
        loss.backward()
        optim.step()
        train_losses.append(loss.detach().to(cpu_device))
        tqdm_obj.set_description_str(f"Epoch {i+1}/{n_epochs} train loss - {loss.detach().to(cpu_device)}")
    
    
    with torch.no_grad():
        transformer.eval()
        src, tar = next(iter(val_dl))
        src, tar = src.to(device), tar.to(device)
        val_logits = transformer(src, tar)
        val_y_pred = torch.argmax(val_logits, dim=-1)
        val_loss = loss_fn(val_logits.permute(0,2,1), tar)
        val_losses.append(val_loss.detach().to(cpu_device))
        
        val_acc, val_em = Acc(val_y_pred, tar).to(cpu_device), EM(val_y_pred, tar).to(cpu_device)
        val_accs.append(val_acc)
        val_ems.append(val_em)
        if val_loss < prev_loss:
            print("Saving Model -", end=" ")
            torch.save(transformer.state_dict(), f"ckpts/test_transformer_{n_train}_{lr}")            
        prev_loss = val_loss
        
    print(f"Val loss: {val_loss.detach().to(cpu_device)} Val acc {val_acc} Val EM {val_em}") 

plt.figure()
plt.plot(val_losses, label="val")
plt.legend()
plt.savefig(f"plots/val_loss_{n_train}_{lr}.pdf")

plt.figure()
plt.plot(train_losses, label="train")
plt.legend()
plt.savefig(f"plots/train_loss_{n_train}_{lr}.pdf")

fig, ax = EM.plot(val_ems)
fig.savefig(f"plots/EM_{n_train}_{lr}.pdf")
fig

fig, ax = Acc.plot(val_accs)
fig.savefig(f"plots/Acc_{n_train}_{lr}.pdf")
fig