import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf

# Step 1: Load and preprocess data
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    return df

def preprocess_data(df):
    # Ensure 'threat_name' column is of string type
    df['threat_name'] = df['threat_name'].astype(str)
    
    # Tokenize input text
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    encoded_data = tokenizer(df['threat_name'].tolist(), truncation=True, padding=True, return_tensors='tf')

    # Encode labels
    labels = df['threat_type'].astype('category').cat.codes

    return encoded_data, labels

# Step 2: Define Transformer model
def build_model(num_labels):
    model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
    return model

# Step 3: Train the model
def train_model(model, train_data, train_labels, val_data, val_labels, batch_size=16, epochs=3):
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])
    
    history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels),
                        batch_size=batch_size, epochs=epochs)
    
    return model, history

# Step 4: Evaluation
def evaluate_model(model, test_data, test_labels):
    results = model.evaluate(test_data, test_labels)
    return results

# Step 5: Example usage
if __name__ == "__main__":
    csv_file = 'mitre_attack_mobile_threats.csv'
    df = load_data(csv_file)
    
    # Preprocess data
    encoded_data, labels = preprocess_data(df)
    
    # Check the shapes of the encoded data and labels
    print(f"Encoded data shape: {encoded_data['input_ids'].shape}")
    print(f"Labels shape: {labels.shape}")

    # Ensure consistent lengths
    if encoded_data['input_ids'].shape[0] == labels.shape[0]:
        # Convert tensors to numpy arrays
        input_ids_np = encoded_data['input_ids'].numpy()
        labels_np = labels.to_numpy()

        # Split data into train, validation, test sets
        train_data, test_data, train_labels, test_labels = train_test_split(
            input_ids_np, labels_np, test_size=0.2, random_state=42
        )
        train_data, val_data, train_labels, val_labels = train_test_split(
            train_data, train_labels, test_size=0.1, random_state=42
        )
        
        # Convert numpy arrays back to tensors
        train_data = tf.convert_to_tensor(train_data)
        val_data = tf.convert_to_tensor(val_data)
        test_data = tf.convert_to_tensor(test_data)

        train_labels = tf.convert_to_tensor(train_labels)
        val_labels = tf.convert_to_tensor(val_labels)
        test_labels = tf.convert_to_tensor(test_labels)

        # Build model
        num_labels = len(df['threat_type'].unique())
        model = build_model(num_labels)
        
        # Train model
        model, history = train_model(model, train_data, train_labels, val_data, val_labels)
        
        # Evaluate model
        test_results = evaluate_model(model, test_data, test_labels)
        print("Test accuracy:", test_results[1])
    else:
        print("Inconsistent lengths between encoded data and labels.")
