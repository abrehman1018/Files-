This project implements a Spiking Neural Network (SNN) model to classify threat types using rate encoding. The dataset is preprocessed, split, and trained using a custom spiking classifier model.

## File Descriptions

### 1. pre_process.py

This script preprocesses the dataset and converts it into spike trains using rate encoding.

- Dependencies: `torch`, `pandas`, `transformers`, `matplotlib`
- Steps:
  1. Load the dataset from `dataset.csv`.
  2. Extract threat names and types.
  3. Load pre-trained BERT model and tokenizer.
  4. Tokenize and embed threat names.
  5. Convert threat types to one-hot embeddings.
  6. Normalize embeddings.
  7. Convert normalized embeddings to spike trains using rate coding.
  8. Save the spike trains as numpy arrays (`threat_name_spike_train.npy` and `threat_type_spike_train.npy`).

### 2. split_dataset.py

This script splits the dataset into training and testing sets and creates DataLoaders.

- Dependencies: `torch`, `numpy`
- Steps:
  1. Load spike train data from numpy arrays.
  2. Combine spike trains and labels into a TensorDataset.
  3. Split the dataset into training and testing sets (80% train, 20% test).
  4. Create DataLoaders for training and testing.
  5. Save the train and test datasets as PyTorch tensors (`train_dataset.pt` and `test_dataset.pt`).

### 3. model.py

This script defines and trains the spiking classifier model.

- Dependencies: `torch`, `numpy`, `norse.torch`
- Steps:
  1. Define the `SpikingClassifier` class.
  2. Load spike train data and labels from numpy arrays.
  3. Ensure the dimensions of the spike trains match.
  4. Combine threat name and threat type spike trains.
  5. Define the model, loss function, and optimizer.
  6. Split data into training and testing sets.
  7. Train the model over a specified number of epochs.
  8. Validate the model and log accuracy.
  9. Save the trained model to `spiking_classifier.pth`.

## Usage

1. Preprocess the dataset:
   
   python pre_process.py

2. Split the dataset:

   python split_dataset.py

3. Train the model:

   python model.py

## Notes

1. Ensure the dataset file (dataset.csv) is in the same directory as the scripts.
2. The labels file (labels.npy) should be present in the same directory for the split_dataset.py and model.py scripts to function correctly.
3. The threat_type_list in pre_process.py should be updated based on the actual threat types in your dataset.
