import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('dataset.csv')

# Extract threat names and threat types
threat_names = data['threat name'].tolist()
threat_types = data['threat type'].tolist()

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize and embed threat names
inputs = tokenizer(threat_names, padding=True, truncation=True, return_tensors='pt')

# Get embeddings
with torch.no_grad():
    threat_name_embeddings = model(**inputs).last_hidden_state

# Define the threat types
threat_type_list = ["x-mitre-matrix", "course-of-action", "malware", "tool", "x-mitre-tactic", "attack-pattern", "x-mitre-data-component", "campaign", "intrusion-set", "x-mitre-data-source"]

# Map threat types to indices
threat_type_to_idx = {threat_type: idx for idx, threat_type in enumerate(threat_type_list)}

# Convert threat types to indices
threat_type_indices = [threat_type_to_idx[threat_type] for threat_type in threat_types]

# Convert indices to one-hot embeddings
threat_type_embeddings = torch.nn.functional.one_hot(torch.tensor(threat_type_indices), num_classes=len(threat_type_list)).float()

# Normalize embeddings
def normalize_embeddings(embeddings):
    min_val = embeddings.min()
    max_val = embeddings.max()
    normalized_embeddings = (embeddings - min_val) / (max_val - min_val)
    return normalized_embeddings

normalized_threat_name_embeddings = normalize_embeddings(threat_name_embeddings)
normalized_threat_type_embeddings = normalize_embeddings(threat_type_embeddings)

# Convert to spike trains (Rate Coding)
def rate_coding(embeddings, spike_rate=100, duration=1.0):
    num_steps = int(spike_rate * duration)
    if len(embeddings.shape) == 3:
        spike_train = torch.rand(embeddings.size(0), embeddings.size(1), embeddings.size(2), num_steps) < embeddings.unsqueeze(3)
    elif len(embeddings.shape) == 2:
        spike_train = torch.rand(embeddings.size(0), embeddings.size(1), num_steps) < embeddings.unsqueeze(2)
    return spike_train.float()

threat_name_spike_train = rate_coding(normalized_threat_name_embeddings)
threat_type_spike_train = rate_coding(normalized_threat_type_embeddings)

# Check the shapes of the spike trains
print(f"Threat Name Spike Train Shape: {threat_name_spike_train.shape}")
print(f"Threat Type Spike Train Shape: {threat_type_spike_train.shape}")

import numpy as np

# Save the spike trains as numpy arrays
np.save('threat_name_spike_train.npy', threat_name_spike_train.cpu().numpy())
np.save('threat_type_spike_train.npy', threat_type_spike_train.cpu().numpy())
