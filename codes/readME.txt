This project involves training and fine-tuning Transformer models, BERT and SpikeFormer, for the classification of threat data. The threat dataset consists of various types of threats, each categorized by its name and type. The main scripts in this project are designed to load, preprocess, train, fine-tune, and evaluate these models using the provided dataset.

Files Overview

1. fine_tune_bert_for_single_sentence.py
This script fine-tunes a BERT model for single sentence classification tasks. It loads a pre-trained BERT model and fine-tunes it on a custom threat dataset. The dataset is split into training, validation, and test sets, and the script supports evaluation using accuracy (acc) or Matthews correlation coefficient (mcc).
Details:
•	Libraries Used: PyTorch, Transformers, pandas, torchmetrics
•	Functions and Classes:
o	ThreatDataset: Custom dataset class for loading and processing threat data.
o	to_device: Helper function to move data to the specified device (CPU/GPU).
o	args: Function to parse command line arguments for various hyperparameters and configurations.
o	fine_tune_teacher_model: Main function to fine-tune the BERT model on the threat dataset.
Usage:
1.	Prepare Data:
o	Ensure you have a CSV file containing the threat data with columns threat_name and threat_type.
o	Modify the data_path in the ThreatDataset initialization to point to your CSV file.
2.	Run the Script:
o	Use the command line to run the script with desired arguments.
o	Example: python fine_tune_bert_for_single_sentence.py --dataset_name cola --epochs 4
Arguments:
•	--dataset_name: Name of the dataset (default: "cola").
•	--batch_size: Batch size for training (default: 50).
•	--fine_tune_lr: Learning rate for fine-tuning (default: 5e-5).
•	--epochs: Number of epochs for training (default: 4).
•	--teacher_model_name: Name of the pre-trained BERT model (default: "bert-base-cased").
•	--label_num: Number of unique threat types (default: 2).
•	--metric: Evaluation metric, either "acc" for accuracy or "mcc" for Matthews correlation coefficient (default: "acc").
Functionality:
•	Data Preparation: The script loads the threat dataset from a CSV file and splits it into training, validation, and test sets.
•	Model Initialization: A pre-trained BERT model is loaded and set up for sequence classification. The model parameters are optimized using AdamW.
•	Training Loop: The training loop involves:
o	Tokenizing the input data.
o	Forward pass through the BERT model.
o	Calculating and printing the cross-entropy loss.
o	Backward pass and optimization step.
o	Resetting the spiking neurons after each batch.
•	Evaluation: The model is evaluated on the test set using either accuracy or Matthews correlation coefficient.
•	Model Saving: The fine-tuned model and tokenizer are saved at the end of each epoch.
Dataset: The dataset should be in a CSV file format with the following columns:
•	threat_name: The name of the threat.
•	threat_type: The type/category of the threat.
Modify the data_path in the script to point to your dataset file.




2. new_distill_spikformer.py
This script implements the distillation process for training a student model using the SpikeFormer architecture, leveraging a pre-trained teacher model based on BERT. The process involves transferring knowledge from the teacher model to the student model, specifically focusing on threat classification tasks.

Details
Libraries Used: PyTorch, Transformers, torchmetrics
Functions and Classes:
to_device: Helper function to move data to the specified device (CPU/GPU).
args: Function to parse command line arguments for various hyperparameters and configurations.
distill: Main function to distill knowledge from the BERT model to the SpikeFormer model, including data loading, tokenization, training, and evaluation.
Usage
1. Prepare Data
Ensure you have a CSV file containing the threat data with columns threat_name and threat_type.
Modify the data_path argument in the script to point to your CSV file.
2. Run the Script
Use the command line to run the script with desired arguments.
Example: python new_distill_spikformer.py --dataset_name "cola" --epochs 4
Arguments
--dataset_name: Name of the dataset to be used (default: "dataset").
--batch_size: Batch size for training (default: 4).
--fine_tune_lr: Learning rate for fine-tuning (default: 1e-2).
--epochs: Number of epochs for training (default: 100).
--teacher_model_path: Path to the pre-trained BERT model.
--label_num: Number of unique threat types (default: 10).
--depths: Number of layers in the student model (default: 6).
--max_length: Maximum length of input sequences (default: 64).
--dim: Dimensionality of the model (default: 768).
--ce_weight: Weight for cross-entropy loss (default: 0.0).
--emb_weight: Weight for embedding loss (default: 1.0).
--logit_weight: Weight for logit loss (default: 1.0).
--rep_weight: Weight for representation loss (default: 5.0).
--num_step: Number of steps for SpikeFormer (default: 32).
--tau: Time constant for SpikeFormer (default: 10.0).
--common_thr: Common threshold for SpikeFormer (default: 1.0).
--predistill_model_path: Path to the pre-distilled model.
--ignored_layers: Number of ignored layers in the teacher model (default: 1).
--metric: Metric to evaluate the model, either accuracy ("acc") or Matthews correlation coefficient ("mcc") (default: "acc").
Functionality
Data Preparation: The script loads the threat dataset from a CSV file and splits it into training, validation, and test sets.
Model Training: The SpikeFormer model is trained using the training data. The training loop involves tokenizing the input data, performing forward passes, calculating the loss, and updating model parameters.
Evaluation: After training, the model's performance is evaluated on the test set using the specified metric (accuracy or Matthews correlation coefficient).
Model Saving: The trained model is saved after each epoch, along with the current timestamp and evaluation metric.



3. predistill_spikformer.py
This script performs pre-distillation training for the SpikFormer model using a teacher-student approach. The teacher model is a pre-trained BERT, and the student model is the SpikFormer. The training involves transferring knowledge from the teacher to the student by matching their representations.
Details:
•	Libraries Used: PyTorch, Transformers, spikingjelly, pandas
•	Functions and Classes:
o	ThreatDataset: Custom dataset class for loading and processing threat data.
o	to_device: Helper function to move data to the specified device (CPU/GPU).
o	args: Function to parse command line arguments for various hyperparameters and configurations.
o	train: Main function to train the SpikFormer model using the pre-distillation method.
Usage:
1.	Prepare Data:
o	Ensure you have a CSV file containing the threat data with columns threat_name and threat_type.
o	Modify the data_path argument in the script to point to your CSV file.
2.	Run the Script:
o	Use the command line to run the script with desired arguments.
o	Example: python predistill_spikformer.py --epochs 1
Arguments:
•	--seed: Random seed for reproducibility (default: 42).
•	--batch_size: Batch size for training (default: 32).
•	--fine_tune_lr: Learning rate for fine-tuning (default: 6e-4).
•	--max_sample_num: Maximum number of samples for training (default: 2e7).
•	--epochs: Number of epochs for training (default: 1).
•	--label_num: Number of unique threat types (default: 10).
•	--depths: Number of layers in the SpikFormer model (default: 12).
•	--max_length: Maximum length for tokenized input sequences (default: 256).
•	--dim: Dimensionality of the model (default: 768).
•	--rep_weight: Weight for the representation loss (default: 0.1).
•	--tau: Time constant for the spiking neuron model (default: 10.0).
•	--common_thr: Common threshold for spiking neurons (default: 1.0).
•	--num_step: Number of time steps for the spiking neuron model (default: 16).
•	--teacher_model_path: Path to the pre-trained BERT teacher model (default: "google-bert/bert-base-cased").
•	--ignored_layers: Number of layers to ignore in the representation matching (default: 0).
•	--data_path: Path to the threat data CSV file (default: "D:/transformer/SpikeBERT/dataset.csv").
Functionality:
•	Data Preparation: The script loads the threat dataset from a CSV file and initializes the data loader for training.
•	Teacher Model Initialization: A pre-trained BERT model is loaded and set to evaluation mode. Its parameters are frozen during training.
•	Student Model Initialization: The SpikFormer model is initialized with specified hyperparameters, including number of layers, dimensions, and spiking neuron parameters.
•	Training Loop: The training loop involves:
o	Tokenizing the input data.
o	Obtaining representations from both the teacher and student models.
o	Calculating the representation loss by comparing the student and teacher outputs.
o	Updating the student model parameters using gradient scaling and optimization.
o	Resetting the spiking neurons after each batch.
•	Model Saving: The trained student model is saved at the end of training.



4. train_spikformer.py
This script trains the SpikFormer model for threat classification tasks. It includes setting up the dataset, initializing the SpikFormer model, training the model, and evaluating its performance.
Details:
•	Libraries Used: PyTorch, Transformers, spikingjelly, pandas
•	Functions and Classes:
o	ThreatDataset: Custom dataset class for loading and processing threat data.
o	to_device: Helper function to move data to the specified device (CPU/GPU).
o	args: Function to parse command line arguments for various hyperparameters and configurations.
o	train: Main function to train the SpikFormer model, including data loading, tokenization, training, and evaluation.
Usage:
1.	Prepare Data:
o	Ensure you have a CSV file containing the threat data with columns threat_name and threat_type.
o	Modify the data_path argument in the script to point to your CSV file.
2.	Run the Script:
o	Use the command line to run the script with desired arguments.
o	Example: python train_spikformer.py --dataset_name "cola" --epochs 100
Arguments:
•	--seed: Random seed for reproducibility (default: 42).
•	--batch_size: Batch size for training (default: 32).
•	--fine_tune_lr: Learning rate for fine-tuning (default: 6e-4).
•	--epochs: Number of epochs for training (default: 100).
•	--label_num: Number of unique threat types (default: 10).
•	--depths: Number of layers in the SpikFormer model (default: 6).
•	--max_length: Maximum length for tokenized input sequences (default: 256).
•	--dim: Dimensionality of the model (default: 768).
•	--tau: Time constant for the spiking neuron model (default: 10.0).
•	--common_thr: Common threshold for spiking neurons (default: 1.0).
•	--num_step: Number of time steps for the spiking neuron model (default: 32).
•	--tokenizer_path: Path to the pre-trained BERT tokenizer (default: "bert-base-cased").
•	--data_path: Path to the threat data CSV file (default: "D:\transformer\SpikeBERT\dataset.csv").
Functionality:
•	Data Preparation: The script loads the threat dataset from a CSV file and initializes the data loader for training.
•	Model Initialization: The SpikFormer model is initialized with specified hyperparameters, including number of layers, dimensions, and spiking neuron parameters.
•	Model Training: The training loop involves tokenizing the input data, performing forward passes, calculating the loss, and updating model parameters using gradient scaling and optimization.
•	Evaluation: After each epoch, the model's performance is evaluated on the training set, and the accuracy is calculated.
•	Model Saving: The trained model is saved if it achieves the best accuracy during training.
