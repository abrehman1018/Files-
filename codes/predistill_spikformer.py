import torch
import torch.nn as nn
import pickle
import argparse
import torch.nn.functional as F
import torch.optim as optim
from model import new_spikformer
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import time
from transformers import BertTokenizer, BertModel
from spikingjelly.activation_based import functional
import math
from utils.public import set_seed
import pandas as pd
from torch.utils.data import Dataset
import random

print(torch.__version__)

class ThreatDataset(Dataset):
    def __init__(self, data_path: str, threat_types: list):
        self.data = pd.read_csv(data_path)
        self.threat_type_to_idx = {threat: idx for idx, threat in enumerate(threat_types)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        row = self.data.iloc[index]
        threat_name = row['threat_name']
        threat_type = self.threat_type_to_idx[row['threat_type']]
        return threat_name, threat_type

def to_device(x, device):
    for key in x:
        x[key] = x[key].to(device)

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--fine_tune_lr", default=6e-4, type=float)
    parser.add_argument("--max_sample_num", default=2e7, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--label_num", default=10, type=int)  # Adjust based on the number of unique threat types
    parser.add_argument("--depths", default=12, type=int)
    parser.add_argument("--max_length", default=256, type=int)
    parser.add_argument("--dim", default=768, type=int)
    parser.add_argument("--rep_weight", default=0.1, type=float)
    parser.add_argument("--tau", default=10.0, type=float)
    parser.add_argument("--common_thr", default=1.0, type=float)
    parser.add_argument("--num_step", default=16, type=int)
    parser.add_argument("--teacher_model_path", default="google-bert/bert-base-cased", type=str)
    parser.add_argument("--ignored_layers", default=0, type=int)
    parser.add_argument("--data_path", default="D:/transformer/SpikeBERT/dataset.csv", type=str)  # Path to your threat data CSV file
    args = parser.parse_args()
    return args

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(args.teacher_model_path)
    teacher_model = BertModel.from_pretrained(args.teacher_model_path, num_labels=args.label_num, output_hidden_states=True).to(device)
    
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()

    student_model = new_spikformer(depths=args.depths, length=args.max_length, T=args.num_step, \
        tau=args.tau, common_thr=args.common_thr, vocab_size=len(tokenizer), dim=args.dim, num_classes=args.label_num, mode="pre_distill")
    
    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.AdamW(params=student_model.parameters(), lr=args.fine_tune_lr)

    threat_types = ["x-mitre-matrix", "course-of-action", "malware", "tool", "x-mitre-tactic", "attack-pattern", "x-mitre-data-component", "campaign", "intrusion-set", "x-mitre-data-source"]  # List of your threat types
    threat_dataset = ThreatDataset(data_path=args.data_path, threat_types=threat_types)
    
    train_data_loader = DataLoader(dataset=threat_dataset, \
        batch_size=args.batch_size, shuffle=True, drop_last=False)

    device_ids = [i for i in range(torch.cuda.device_count())]
    print(device_ids)
    if len(device_ids) > 1:
        student_model = nn.DataParallel(student_model, device_ids=device_ids).to(device)
    student_model = student_model.to(device)

    train_iter = 0
    skip_p = args.max_sample_num / len(threat_dataset)
    print(f"all samples: {len(threat_dataset)}, skip_p: {skip_p}")

    for batch in tqdm(train_data_loader):
        p = random.uniform(0, 1)
        if p > skip_p:
            continue
        train_iter += 1
        threat_names, threat_types = batch
        student_model.train()
        inputs = tokenizer(threat_names, padding="max_length", truncation=True, \
            return_tensors="pt", max_length=args.max_length)
        to_device(inputs, device)
        threat_types = threat_types.to(device)
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)
        
        tea_rep = teacher_outputs.hidden_states[1:][::int(12/args.depths)]  # layers output
        stu_rep, student_outputs = student_model(inputs['input_ids'])

        student_outputs = student_outputs.reshape(-1, args.num_step, args.label_num)

        student_outputs = student_outputs.transpose(0, 1)

        tea_rep = torch.tensor(np.array([item.cpu().detach().numpy() for item in tea_rep]), dtype=torch.float32)
        tea_rep = tea_rep.to(device=device)
        
        rep_loss = 0
        tea_rep = tea_rep[args.ignored_layers:]
        stu_rep = stu_rep[args.ignored_layers:]
        for i in range(len(stu_rep)):
            rep_loss += F.mse_loss(stu_rep[i], tea_rep[i])
        rep_loss = rep_loss / len(threat_names)  # batch mean

        total_loss = args.rep_weight * rep_loss

        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        functional.reset_net(student_model)

        print(f"In iter {train_iter}, rep_loss: {rep_loss}, total_loss: {total_loss}")

    torch.save(student_model.state_dict(), \
        f"saved_models/predistill_spikformer/" + f"_lr{args.fine_tune_lr}_seed{args.seed}" + 
        f"_batch_size{args.batch_size}_depths{args.depths}_max_length{args.max_length}" + 
        f"_tau{args.tau}_common_thr{args.common_thr}"
    )  
    return

if __name__ == "__main__":
    _args = args()
    for arg in vars(_args):
        print(arg, getattr(_args, arg))
    set_seed(_args.seed)
    train(_args)
