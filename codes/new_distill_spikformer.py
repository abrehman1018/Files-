import torch
import os
import torch.nn as nn
import pickle
import argparse
import torch.nn.functional as F
import torch.optim as optim
from model import new_spikformer
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import os
import time
from dataset import ThreatDataset  # Update this import
from transformers import BertTokenizer, BertForSequenceClassification
from spikingjelly.activation_based import encoding, functional
import math
from utils.public import set_seed
from torchmetrics.classification import MatthewsCorrCoef

os.environ["CUDA_VISIBLE_DEVICES"] = "1,0,2,3"

def to_device(x, device):
    for key in x:
        x[key] = x[key].to(device)
    

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--dataset_name", default="dataset", type=str)  # Update dataset name
    parser.add_argument("--data_augment", default="True", type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--fine_tune_lr", default=1e-2, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--teacher_model_path", default="", type=str)
    parser.add_argument("--label_num", default=10, type=int)  # Update label number based on your dataset
    parser.add_argument("--depths", default=6, type=int)
    parser.add_argument("--max_length", default=64, type=int)
    parser.add_argument("--dim", default=768, type=int)
    parser.add_argument("--ce_weight", default=0.0, type=float)
    parser.add_argument("--emb_weight", default=1.0, type=float)
    parser.add_argument("--logit_weight", default=1.0, type=float)
    parser.add_argument("--rep_weight", default=5.0, type=float)
    parser.add_argument("--num_step", default=32, type=int)
    parser.add_argument("--tau", default=10.0, type=float)
    parser.add_argument("--common_thr", default=1.0, type=float)
    parser.add_argument("--predistill_model_path", default="", type=str)
    parser.add_argument("--ignored_layers", default=1, type=int)
    parser.add_argument("--metric", default="acc", type=str)
    args = parser.parse_args()
    return args

def distill(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(args.teacher_model_path)
    teacher_model = BertForSequenceClassification.from_pretrained(args.teacher_model_path, num_labels=args.label_num, output_hidden_states=True).to(device)
    
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()

    student_model = new_spikformer(depths=args.depths, length=args.max_length, T=args.num_step, 
                                   tau=args.tau, common_thr=args.common_thr, vocab_size=len(tokenizer), dim=args.dim, num_classes=args.label_num, mode="distill")
    
    if args.predistill_model_path != "":
        weights = torch.load(args.predistill_model_path)
        student_model.load_state_dict(weights, strict=False)
        print("load predistill model finish!")
    
    scaler = torch.cuda.amp.GradScaler()
    optimer = torch.optim.AdamW(params=student_model.parameters(), lr=args.fine_tune_lr, betas=(0.9, 0.999), weight_decay=5e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimer, T_max=args.epochs, eta_min=0)
    
    threat_types = ["x-mitre-matrix", "course-of-action", "malware", "tool", "x-mitre-tactic", "attack-pattern", "x-mitre-data-component", "campaign", "intrusion-set", "x-mitre-data-source"]

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
        student_model = nn.DataParallel(student_model, device_ids=device_ids).to(device)
    student_model = student_model.to(device)

    metric_list = []
    for epoch in tqdm(range(args.epochs)):
        total_loss_list = []
        embeddings_loss_list = []
        ce_loss_list = []
        logit_loss_list = []
        rep_loss_list = []
        for batch in tqdm(train_data_loader):
            student_model.train()
            batch_size = len(batch[0])
            labels = batch[1].to(device)
            inputs = tokenizer(batch[0], padding="max_length", truncation=True, 
                               return_tensors="pt", max_length=args.max_length)
            
            to_device(inputs, device)
            with torch.no_grad():
                teacher_outputs = teacher_model(**inputs)
            tea_embeddings = teacher_model.bert.embeddings.word_embeddings.weight
            if len(device_ids) > 1:
                stu_embeddings = student_model.module.emb.weight
            else:
                stu_embeddings = student_model.emb.weight

            embeddings_loss = F.mse_loss(stu_embeddings, tea_embeddings)
            embeddings_loss_list.append(embeddings_loss.item())

            tea_rep = teacher_outputs.hidden_states[1:][::int(12/args.depths)]
            stu_rep, student_outputs = student_model(inputs['input_ids'])

            student_outputs = student_outputs.reshape(-1, args.num_step, args.label_num)
            student_outputs = student_outputs.transpose(0, 1)
            student_logits = torch.mean(student_outputs, dim=0)

            ce_loss = F.cross_entropy(student_logits, labels)
            ce_loss_list.append(ce_loss.item())

            logit_loss = F.kl_div(F.log_softmax(student_logits, dim=1), F.softmax(teacher_outputs.logits, dim=1), reduction='batchmean')
            logit_loss_list.append(logit_loss.item())

            tea_rep = torch.tensor(np.array([item.cpu().detach().numpy() for item in tea_rep]), dtype=torch.float32)
            tea_rep = tea_rep.to(device=device)
            
            rep_loss = 0
            tea_rep = tea_rep[args.ignored_layers:]
            stu_rep = stu_rep[args.ignored_layers:]
            for i in range(len(stu_rep)):
                rep_loss += F.mse_loss(stu_rep[i], tea_rep[i])
            rep_loss = rep_loss / batch_size
            rep_loss_list.append(rep_loss.item())

            total_loss = (args.emb_weight * embeddings_loss) + (args.ce_weight * ce_loss) + (args.logit_weight * logit_loss) + (args.rep_weight * rep_loss)
            total_loss_list.append(total_loss.item())

            optimer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimer)
            scaler.update()
            functional.reset_net(student_model)

        scheduler.step()

        y_true = []
        y_pred = []
        student_model.eval()
        with torch.no_grad():
            for batch in tqdm(test_data_loader):
                batch_size = len(batch[0])
                b_y = batch[1]
                y_true.extend(b_y.to("cpu").tolist())
                inputs = tokenizer(batch[0], padding="max_length", truncation=True, return_tensors='pt', max_length=args.max_length)
                to_device(inputs, device)
                _, outputs = student_model(inputs['input_ids'])
                outputs = outputs.to("cpu")
                outputs = outputs.reshape(-1, args.num_step, args.label_num)
                outputs = outputs.transpose(0, 1)
                logits = torch.mean(outputs, dim=0)
                y_pred.extend(torch.max(logits, 1)[1].tolist())

                functional.reset_net(student_model)

        if args.metric == "acc":
            correct = 0
            for i in range(len(y_true)):
                correct += 1 if y_true[i] == y_pred[i] else 0
            acc = correct / len(y_pred)
            print("acc", acc)
        elif args.metric == "mcc":
            print(y_true)
            print(y_pred)
            matthews_corrcoef = MatthewsCorrCoef(task='binary')
            mcc = matthews_corrcoef(torch.tensor(y_true), torch.tensor(y_pred))
            print("mcc, ", mcc)

        record = acc if args.metric == "acc" else mcc
        metric_list.append(record)
        print(f"Epoch {epoch} {args.metric}: {record}")
        if record >= np.max(metric_list):
            torch.save(student_model.state_dict(),
                    f"saved_models/distilled_spikformer/{args.dataset_name}_epoch{epoch}_{record}" + 
                    f"num_step_{args.num_step}_lr{args.fine_tune_lr}_seed{args.seed}" + 
                    f"_batch_size{args.batch_size}_depths{args.depths}_max_length{args.max_length}" + 
                    f"_ce_weight{args.ce_weight}_logit_weight{args.logit_weight}_rep_weight{args.rep_weight}" +
                    f"_tau{args.tau}_common_thr{args.common_thr}"
                    )
    print("best: ", np.max(metric_list))
    return


if __name__ == "__main__":
    _args = args()
    for arg in vars(_args):
        print(arg, getattr(_args, arg))
    set_seed(_args.seed)
    distill(_args)
