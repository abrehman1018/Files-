import numpy as np
import torch
import torch.nn as nn
import norse.torch as snn
from torch.utils.data import DataLoader, TensorDataset, random_split

class SpikingClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SpikingClassifier, self).__init__()
        self.hidden = snn.LIFRecurrentCell(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        seq_length, batch_size, _ = x.size()
        hidden_state = None
        for t in range(seq_length):
            hidden_state, _ = self.hidden(x[t], hidden_state)
        out = self.output(hidden_state)
        return out

# Load spike train data
threat_name_spike_train = torch.from_numpy(np.load('threat_name_spike_train.npy'))
threat_type_spike_train = torch.from_numpy(np.load('threat_type_spike_train.npy'))
labels = torch.from_numpy(np.load('labels.npy'))

# Check dimensions of spike trains and labels
print(f"Threat name spike train shape: {threat_name_spike_train.shape}")
print(f"Threat type spike train shape: {threat_type_spike_train.shape}")
print(f"Labels shape: {labels.shape}")

# Ensure both tensors have the same dimensions
# Expand threat_type_spike_train to match the shape of threat_name_spike_train
threat_type_spike_train = threat_type_spike_train.unsqueeze(1).expand(-1, 14, -1, -1)

# Combine the two spike trains
combined_spike_train = torch.cat((threat_name_spike_train, threat_type_spike_train), dim=2)

# Check if the labels match the combined_spike_train along the batch dimension
if combined_spike_train.size(0) != labels.size(0):
    raise ValueError("Mismatch between combined_spike_train and labels batch sizes")

# Define your model, loss function, and optimizer
input_size = combined_spike_train.shape[2]
model = SpikingClassifier(input_size=input_size, hidden_size=128, output_size=10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Split data into training and test sets
train_size = int(0.8 * combined_spike_train.size(0))
test_size = combined_spike_train.size(0) - train_size
dataset = TensorDataset(combined_spike_train, labels)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    for threat_spikes, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(threat_spikes)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation and logging
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for threat_spikes, labels in test_loader:
            outputs = model(threat_spikes)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%')

# Save the trained model
torch.save(model.state_dict(), 'spiking_classifier.pth') 

