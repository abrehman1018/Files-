import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F
import torch.optim as optim
from model import new_spikformer
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import time
from transformers import BertTokenizer
from spikingjelly.activation_based import functional
from utils.public import set_seed
import pandas as pd
from torch.utils.data import Dataset

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
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--label_num", default=10, type=int)  # Number of unique threat types
    parser.add_argument("--depths", default=6, type=int)
    parser.add_argument("--max_length", default=256, type=int)
    parser.add_argument("--dim", default=768, type=int)
    parser.add_argument("--tau", default=10.0, type=float)
    parser.add_argument("--common_thr", default=1.0, type=float)
    parser.add_argument("--num_step", default=32, type=int)
    parser.add_argument("--tokenizer_path", default="bert-base-cased", type=str)
    parser.add_argument("--data_path", default="D:\transformer\SpikeBERT\dataset.csv", type=str)  # Path to your threat data CSV file
    args = parser.parse_args()
    return args

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)

    model = new_spikformer(depths=args.depths, length=args.max_length, T=args.num_step,
                           tau=args.tau, common_thr=args.common_thr, vocab_size=len(tokenizer), dim=args.dim, num_classes=args.label_num, mode="distill")

    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.fine_tune_lr, betas=(0.9, 0.999), weight_decay=5e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

    threat_types = ["x-mitre-matrix", "course-of-action", "malware", "tool", "x-mitre-tactic", "attack-pattern", "x-mitre-data-component", "campaign", "intrusion-set", "x-mitre-data-source"]
    train_dataset = ThreatDataset(data_path=args.data_path, threat_types=threat_types)
    
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    
    # You can add test and validation datasets similarly if needed:
    # test_dataset = ThreatDataset(data_path="path_to_test_data.csv", threat_types=threat_types)
    # test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    # valid_dataset = ThreatDataset(data_path="path_to_validation_data.csv", threat_types=threat_types)
    # valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    device_ids = [i for i in range(torch.cuda.device_count())]
    print(device_ids)
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids).to(device)
    model = model.to(device)

    acc_list = []
    for epoch in tqdm(range(args.epochs)):
        avg_loss = []
        for batch in tqdm(train_data_loader):
            batch_size = len(batch[0])
            labels = batch[1].to(device)
            inputs = tokenizer(batch[0], padding="max_length", truncation=True, return_tensors="pt", max_length=args.max_length)
            to_device(inputs, device)
            _, outputs = model(inputs['input_ids'])
            outputs = outputs.reshape(-1, args.num_step, args.label_num)
            outputs = outputs.transpose(0, 1)
            logits = torch.mean(outputs, dim=0)
            loss = F.cross_entropy(logits, labels)
            avg_loss.append(loss.item())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            functional.reset_net(model)

        scheduler.step()
        print(f"avg_loss at epoch {epoch}: {np.mean(avg_loss)}")

        # eval
        result = []
        all = len(train_dataset)
        model.eval()
        with torch.no_grad():
            correct = 0
            for batch in tqdm(train_data_loader):
                batch_size = len(batch[0])
                b_y = batch[1]
                inputs = tokenizer(batch[0], padding="max_length", truncation=True, return_tensors='pt', max_length=args.max_length)
                to_device(inputs, device)
                _, outputs = model(inputs['input_ids'])
                outputs = outputs.to("cpu")
                outputs = outputs.reshape(-1, args.num_step, args.label_num)
                outputs = outputs.transpose(0, 1)
                logits = torch.mean(outputs, dim=0)
                for line in torch.softmax(logits, 1):
                    result.append(line.numpy())
                correct += int(b_y.eq(torch.max(logits, 1)[1]).sum())
                functional.reset_net(model)

        result = np.asarray(result)
        acc = float(correct)/all
        acc_list.append(acc)
        print(f"Epoch {epoch} Acc: {acc}")
        if acc >= np.max(acc_list):
            torch.save(model.state_dict(), f"saved_models/trained_spikformer/threat_classification_epoch{epoch}_{acc}" +
                       f"_lr{args.fine_tune_lr}_seed{args.seed}" +
                       f"_batch_size{args.batch_size}_depths{args.depths}_max_length{args.max_length}" +
                       f"_tau{args.tau}_common_thr{args.common_thr}"
                       )
    print("best acc: ", np.max(acc_list))
    return

if __name__ == "__main__":
    _args = args()
    for arg in vars(_args):
        print(arg, getattr(_args, arg))
    set_seed(_args.seed)
    train(_args)
