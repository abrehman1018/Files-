import torch
import torch.nn as nn
import pickle
import argparse
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import time
from dataset import TxtDataset
from transformers import BertTokenizer, BertForSequenceClassification
from torchmetrics.classification import MatthewsCorrCoef

def to_device(x, device):
    for key in x:
        x[key] = x[key].to(device)

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name",default="cola",type=str)
    parser.add_argument("--batch_size",default=50,type=int)
    parser.add_argument("--fine_tune_lr",default=5e-5,type=float)
    parser.add_argument("--epochs",default=4,type=int)
    parser.add_argument("--teacher_model_name",default="bert-base-cased",type=str)
    parser.add_argument("--label_num",default=2,type=int)
    parser.add_argument("--metric", default="acc", type=str)
    args = parser.parse_args()
    return args

def fine_tune_teacher_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(args.teacher_model_name)
    model = BertForSequenceClassification.from_pretrained(args.teacher_model_name, num_labels=args.label_num)
    optimizer = AdamW(model.parameters(), lr=args.fine_tune_lr)
    
    threat_types = ["x-mitre-matrix", "course-of-action", "malware", "tool", 
                    "x-mitre-tactic", "attack-pattern", "x-mitre-data-component", 
                    "campaign", "intrusion-set", "x-mitre-data-source"]
    dataset = ThreatDataset(data_path=f"D:\transformer\SpikeBERT\dataset.csv", threat_types=threat_types)
    
    # Split dataset into train, validation, and test sets
    train_size = int(0.8 * len(dataset))
    valid_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

    train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    device_ids = [i for i in range(torch.cuda.device_count())]
    print(device_ids)
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids).to(device)
    model = model.to(device=device)
    model.train()
    
    for epoch in tqdm(range(args.epochs)):
        loss_list = []
        for i, batch in enumerate(train_data_loader):
            inputs = tokenizer(batch[0], padding=True, truncation=True, return_tensors="pt", max_length=512)
            labels = torch.tensor(batch[1].tolist(), dtype=torch.long).to(device)
            to_device(inputs, device)
            outputs = model(**inputs)
            loss = F.cross_entropy(outputs.logits, labels)
            loss_list.append(loss.item())
            print(torch.mean(torch.tensor(loss_list)).item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_list = []

        y_true = []
        y_pred = []
        with torch.no_grad():
            model.eval()
            for batch in test_data_loader:
                b_y = batch[1]
                y_true.extend(b_y.to("cpu").tolist())
                input_dict = tokenizer(batch[0], return_tensors='pt', padding=True, truncation=True, max_length=512)
                to_device(input_dict, "cuda")
                output = (model(**input_dict).logits).to("cpu")
                y_pred.extend(torch.max(output, 1)[1].tolist())
        
        if args.metric == "acc":
            correct = 0
            for i in range(len(y_true)):
                correct += 1 if y_true[i] == y_pred[i] else 0
            acc = correct / len(y_pred)
            print("acc", acc)
        elif args.metric == "mcc":
            matthews_corrcoef = MatthewsCorrCoef(task='binary')
            mcc = matthews_corrcoef(torch.tensor(y_true), torch.tensor(y_pred))
            print("mcc, ", mcc)
        
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        record = acc if args.metric == "acc" else mcc
        tokenizer.save_pretrained(f"saved_models/{args.teacher_model_name}_{current_time}_{args.dataset_name}_epoch{epoch}_{record}")
        if len(device_ids) <= 1:
            model.save_pretrained(f"saved_models/{args.teacher_model_name}_{current_time}_{args.dataset_name}_epoch{epoch}_{record}")
        else:
            model.module.save_pretrained(f"saved_models/{args.teacher_model_name}_{current_time}_{args.dataset_name}_epoch{epoch}_{record}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _args = args()
    _args.label_num = len(["x-mitre-matrix", "course-of-action", "malware", "tool", 
                           "x-mitre-tactic", "attack-pattern", "x-mitre-data-component", 
                           "campaign", "intrusion-set", "x-mitre-data-source"])  # Set the number of labels
    fine_tune_teacher_model(_args)
