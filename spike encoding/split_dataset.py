import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

# Load spike train data
threat_name_spike_train = torch.from_numpy(np.load('threat_name_spike_train.npy'))
threat_type_spike_train = torch.from_numpy(np.load('threat_type_spike_train.npy'))
labels = torch.from_numpy(np.load('labels.npy'))

# Combine spike trains and labels into a TensorDataset
dataset = TensorDataset(threat_name_spike_train, threat_type_spike_train, labels)

# Define train/test split ratio
train_ratio = 0.8
train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size

# Split the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Save train and test datasets
torch.save(train_dataset.dataset.tensors, 'train_dataset.pt')
torch.save(test_dataset.dataset.tensors, 'test_dataset.pt')

# Print dataset sizes
print(f'Training set size: {len(train_dataset)}')
print(f'Test set size: {len(test_dataset)}')
