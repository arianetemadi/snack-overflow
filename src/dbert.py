import re
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader,TensorDataset
from tqdm import tqdm  # Standard tqdm for scripts
from tqdm.notebook import tqdm as notebook_tqdm
import torch.nn as nn
import wandb
from transformers import AdamW, get_scheduler
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef,
    classification_report)
from matplotlib import pyplot as plt



def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters (including hyphens)
    text = re.sub(r"-", "", text)  # Remove hyphens
    text = text.strip()  # Remove leading/trailing whitespace
    return text


def extract_data(headlines):
    texts = []
    labels = []

    # Extract text and labels
    for headline in headlines:
        text = headline[0].metadata['text']
        label = int(headline[0].metadata['class'])

        text = clean_text(text)

        texts.append(text)
        labels.append(label)

    return texts, labels



# Function to encode the text for BERT (convert text to input_ids and attention_masks)
def encode_data(texts, tokenizer):

    # Tokenize the text using BERT tokenizer
    encoding = tokenizer.batch_encode_plus(
        texts,
        max_length=16,
        truncation = True,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding=True)

    input_ids = encoding['input_ids']  # Remove batch dimension
    attention_masks = encoding['attention_mask']  # Remove batch dimension

    return torch.tensor(input_ids), torch.tensor(attention_masks)



# Custom Dataset class, to store raw sentences
class SarcasmDataset(Dataset):
    def __init__(self, tokens, masks, labels, sentences):
        self.tokens = tokens
        self.masks = masks
        self.labels = labels
        self.sentences = sentences

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return {
            "input_ids": self.tokens[idx],
            "attention_mask": self.masks[idx],
            "label": self.labels[idx],
            "sentence": self.sentences[idx]
        }


def dataloader(tokens, masks, labels, sentences, batch_size=16,  isTrain = False):

    dataset = SarcasmDataset(tokens, masks, torch.tensor(labels), sentences)

    if isTrain == "True":
        loader = DataLoader(dataset, batch_size = batch_size, num_workers=2, shuffle = True, drop_last=True)
    else:
        loader = DataLoader(dataset, batch_size = batch_size, num_workers=2, shuffle = False, drop_last=True)


    return loader




# Initialize WandB
wandb.init(
    project="sarcasm-detection",
    config={
        "epochs": 10,
        "batch_size": 8,
        "learning_rate":5e-5,
        "scheduler": "linear",
        "loss_function": "BCELoss"
    }
)

# Training function with progress tracking, early stopping, and WandB logging
def train_bert(model, trainloader, validationloader, device, epochs=10, patience=3):
    # Move model to device
    model.to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    # Scheduler
    total_training_steps = len(trainloader) * epochs
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0.1 * total_training_steps, num_training_steps=total_training_steps)

    # Loss function
    loss_function = nn.BCEWithLogitsLoss()

    # Track best validation accuracy and early stopping
    best_validation_accuracy = 0.0
    best_validation_loss = float('inf')
    best_model_state = None
    no_improvement_epochs = 0

    # Metrics for WandB logging
    running_validation_loss = []
    running_training_loss = []
    validation_acc = []
    training_acc = []

    print(f"Starting training for {epochs} epochs with early stopping (patience={patience}).")

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # Training phase
        model.train()
        total_loss = 0
        correct_predictions = 0

        with tqdm(trainloader, desc="Training", unit="batch") as train_progress:

          for batch in train_progress:
              # Extract batch data
              input_ids = batch["input_ids"]
              attention_mask = batch["attention_mask"]
              labels = batch["label"]
              raw_sentences = batch["sentence"]

              inputs, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

              optimizer.zero_grad()
              outputs = model(inputs, attention_mask)

              logits = outputs.logits
              loss = loss_function(logits[:,1], labels.float()) # selecting the logit for the positive class (sarcastic)


              # Compute loss
              loss.backward()

              optimizer.step()
              scheduler.step()

              total_loss += loss.item()

              # Convert logits to probabilities (if using BCEWithLogitsLoss)
              probabilities = torch.sigmoid(logits[:,1]) # selecting the logit for the positive class (sarcastic)


              # Threshold probabilities to obtain binary predictions
              predictions = (probabilities >= 0.5).float()

              # Calculate accuracy
              correct_predictions += (predictions == labels).sum().item()

              # Update progress bar
              train_progress.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(trainloader.dataset)
        avg_train_accuracy = correct_predictions / len(trainloader.dataset)
        running_training_loss.append(avg_train_loss)
        training_acc.append(avg_train_accuracy)

        print(f"Total loss: {total_loss:.4f}, Training Loss: {avg_train_loss:.4f}, Training Accuracy: {avg_train_accuracy:.4f}")

        # Log training metrics to WandB
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_accuracy": avg_train_accuracy
        })

        # Validation phase
        validation_accuracy, validation_loss = validate_bert(model, validationloader, loss_function, device)
        running_validation_loss.append(validation_loss)
        validation_acc.append(validation_accuracy)

        print(f"Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_accuracy:.4f}")

        # Log validation metrics to WandB
        wandb.log({
            "epoch": epoch,
            "val_loss": validation_loss,
            "val_accuracy": validation_accuracy
        })

        # Check for early stopping
        if validation_accuracy > best_validation_accuracy and validation_loss < best_validation_loss:
            best_validation_accuracy = validation_accuracy
            best_validation_loss = validation_loss
            best_model_state = model.state_dict()  # Save the best model state
            no_improvement_epochs = 0
            print("Validation accuracy improved. Best model updated.")
            # Save best model weights to WandB
            torch.save(model.state_dict(), "best_model.pth")
           # torch.save(model.state_dict(),"best_model.pth")
            wandb.save("best_model.pth", policy="end")  
        else:
            no_improvement_epochs += 1
            print(f"No improvement in validation for {no_improvement_epochs} consecutive epoch(s).")

        if no_improvement_epochs >= patience:
            print("Early stopping triggered. Stopping training.")
            break

    # Restore the best model state (if early stopping was triggered)
    if best_model_state:
        model.load_state_dict(best_model_state)

    print("Training completed.")
    return best_validation_accuracy, running_validation_loss, running_training_loss, validation_acc, training_acc


# Validation function
def validate_bert(model, dataloader, loss_function, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0

    with torch.no_grad():

        for batch in tqdm(dataloader, desc="Validation", unit="batch"):
            # Extract batch data
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["label"]
            raw_sentences = batch["sentence"]

            inputs, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            outputs = model(inputs, attention_mask)

            logits = outputs.logits
            loss = loss_function(logits[:,1], labels.float()) # selecting the logit for the positive class (sarcastic)

            # Compute loss
            total_loss += loss.item()

            # Convert logits to probabilities (if using BCEWithLogitsLoss)
            probabilities = torch.sigmoid(logits[:,1])

            # Threshold probabilities to obtain binary predictions
            predictions = (probabilities >= 0.5).float()

            # Calculate accuracy
            correct_predictions += (predictions == labels).sum().item()

    accuracy = correct_predictions / len(dataloader.dataset)
    avg_loss = total_loss / len(dataloader.dataset)

    return accuracy, avg_loss



def test_bert(model, testloader, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_predictions = []
    all_probabilities = []
    sentences = []

    # Loss function
    loss_function = nn.BCEWithLogitsLoss()

    with torch.no_grad():

        for batch in tqdm(testloader, desc="Testing", unit="batch"):
            # Extract batch data
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["label"]
            raw_sentences = batch["sentence"]

            inputs, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)


            # Model outputs probabilities (sigmoid applied)
            outputs = model(inputs, attention_mask)
            logits = outputs.logits
            #loss = loss_function(logits, labels)
            loss = loss_function(logits[:,1], labels.float()) # selecting the logit for the positive class (sarcastic)


            # Compute loss
            total_loss += loss.item()


            # Convert logits to probabilities (if using BCEWithLogitsLoss)
            probabilities = torch.sigmoid(logits[:,1])

            # Threshold probabilities to obtain binary predictions
            predictions = (probabilities >= 0.5).float()

            labels = labels.cpu().numpy()

            all_probabilities.extend(probabilities.cpu().detach().numpy()) #move to cpu and detach from computation graph
            all_predictions.extend(predictions.cpu().detach().numpy()) #move to cpu and detach from computation graph
            all_labels.extend(labels)
            sentences.extend(raw_sentences)

    # Convert lists to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    roc_auc = roc_auc_score(all_labels, all_probabilities)
    mcc = matthews_corrcoef(all_labels, all_predictions)
    cm = confusion_matrix(all_labels, all_predictions)

    # Log results to WandB
    wandb.log({
        "test_loss": total_loss / len(testloader.dataset),
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "test_roc_auc": roc_auc,
        "test_mcc": mcc,
    })

    # Create a DataFrame
    df = pd.DataFrame({
        "True Label": all_labels,
        "Predicted Label": all_predictions,
        "Probability": all_probabilities,
        "Sentences": sentences
    })

    # Print metrics
    print("\nTest Metrics:")
    print(f"Loss: {total_loss / len(testloader.dataset):.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    # Return all metrics
    return {
        "loss": total_loss / len(testloader.dataset),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "mcc": mcc,
        "confusion_matrix": cm,
        "df": df
    }


def plot_metrics(val_loss, train_loss, val_acc, train_acc):
    # Create a figure and axis
    plt.figure(figsize=(10, 4))

    # Plot training and validation loss
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, first subplot
    plt.plot(val_loss, label='Validation Loss', color='blue', linestyle='--', marker='o')
    plt.plot(train_loss, label='Training Loss', color='red', linestyle='-', marker='x')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot training and validation accuracy
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, second subplot
    plt.plot(val_acc, label='Validation Accuracy', color='blue', linestyle='--', marker='o')
    plt.plot(train_acc, label='Training Accuracy', color='red', linestyle='-', marker='x')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Show the plots
    plt.tight_layout()
    plt.show()


