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

train_dataset = load_dataset('cfilt/iitb-english-hindi', split="train[:36000]")
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

# load tokenizer
# tokenizer = Tokenizer.from_file(str(tokenizer_path))

# Apply tokenizer & make vocab
for lang in ["hi", "en"]:
    tokenizer_path = f"dataset/vocab_{lang}.json"
    if os.path.exists(tokenizer_path):
        # load tokenizer
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['<unk>', '<pad>', '<sos>', '<eos>'], min_frequency=2)
        tokenizer.train_from_iterator(tr['translation'][lang] + vl['translation'][lang]+ tt['translation'][lang], trainer=trainer)

        # save tokenizer
        tokenizer.save(f"dataset/vocab_{lang}.json")


tokenizer_src = Tokenizer.from_file(f"dataset/vocab_en.json")
tokenizer_tar = Tokenizer.from_file(f"dataset/vocab_hi.json")

ds = NMTDataset(tokenizer_src, tokenizer_tar, train_dataloader)
print(len(ds))

B = 64
dl = DataLoader(ds, batch_size=B)
n = 50
d_model = 512
n_h = 8
d_k = d_model/n_h
transformer = Transformer(tokenizer_src.get_vocab_size(), tokenizer_tar.get_vocab_size(), d_model, n_h, 6, 2048, 100, 0.1)

x = next(iter(dl))
op = transformer(x[0], x[1])
op