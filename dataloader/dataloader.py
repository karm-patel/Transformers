import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.nn.utils.rnn import pad_sequence
gpu_device = torch.device("cuda")

DATASET_KEY = 'cfilt/iitb-english-hindi'

class NMTDataset(Dataset):
    def __init__(self, tokenizer_src, tokenizer_tar, dataloader, max_length=100):
        self.data = []
        self.src_tokens, self.tar_tokens = [], []
        x = next(iter(dataloader))["translation"]
        for src_sent, tar_sent in zip(x["en"], x["hi"]):
            src_tokens  = [tokenizer_src.token_to_id("<sos>")] + tokenizer_src.encode(src_sent).ids + [tokenizer_src.token_to_id("<eos>")]
            src_tokens = torch.tensor(src_tokens + [tokenizer_src.token_to_id("<pad>")]*(max_length-len(src_tokens)))
            
            tar_tokens = [tokenizer_tar.token_to_id("<sos>")] + tokenizer_tar.encode(tar_sent).ids + [tokenizer_tar.token_to_id("<eos>")]
            tar_tokens = torch.tensor(tar_tokens + [tokenizer_src.token_to_id("<pad>")]*(max_length-len(tar_tokens)))

            self.src_tokens.append(src_tokens)
            self.tar_tokens.append(tar_tokens)

        # self.src_tokens = pad_sequence(self.src_tokens, padding_value=tokenizer_src.token_to_id("<pad>"))
        # self.tar_tokens = pad_sequence(self.tar_tokens, padding_value=tokenizer_tar.token_to_id("<pad>"))

        
    def __len__(self):
        return len(self.src_tokens)

    def __getitem__(self, idx):
        return self.src_tokens[idx], self.tar_tokens[idx]
