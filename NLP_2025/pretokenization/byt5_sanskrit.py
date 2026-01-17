import os
import pandas as pd
import numpy as np
import torch

# Disable torch.compile to avoid triton issues
torch._dynamo.config.suppress_errors = True
os.environ['TORCH_COMPILE_DISABLE'] = '1'

from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime

# Set device - Force CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️ WARNING: CUDA not available, using CPU (training will be slow)")

# Configuration
class Config:
    model_name = "google/byt5-small"
    max_length = 128
    batch_size = 32  # Increased from 16 (better GPU utilization)
    learning_rate = 5e-5
    num_epochs = 30  # Increased from 10 (better convergence)
    warmup_steps = 1000  # Increased proportionally with epochs
    gradient_accumulation_steps = 2  # Effective batch size = 64
    max_grad_norm = 1.0
    save_dir = "byt5_best_model"
    results_dir = "training_results"
    
config = Config()

# Create directories
os.makedirs(config.save_dir, exist_ok=True)
os.makedirs(config.results_dir, exist_ok=True)

# Custom Dataset
class WordSplitDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length):
        self.data = pd.read_csv(csv_path)
        # Clean data: remove rows with missing values
        self.data = self.data.dropna(subset=['Word', 'Split'])
        # Convert to string to handle any non-string values
        self.data['Word'] = self.data['Word'].astype(str)
        self.data['Split'] = self.data['Split'].astype(str)
        # Reset index after dropping rows
        self.data = self.data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"Loaded {len(self.data)} valid samples from {csv_path}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        input_text = str(row['Word']).strip()
        target_text = str(row['Split']).strip()
        
        # Skip empty strings
        if not input_text or not target_text:
            # Return a default valid sample if empty
            input_text = "default"
            target_text = "default"
        
        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels = target_encoding['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': labels
        }

# Metrics calculation
def calculate_metrics(predictions, labels, tokenizer):
    # Decode predictions and labels
    pred_strs = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    label_strs = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Calculate exact match accuracy
    correct = sum([1 for pred, label in zip(pred_strs, label_strs) if pred.strip() == label.strip()])
    accuracy = correct / len(pred_strs)
    
    return accuracy, pred_strs, label_strs

# Training function
def train_epoch(model, dataloader, optimizer, scheduler, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss = loss / config.gradient_accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * config.gradient_accumulation_steps
        progress_bar.set_postfix({'loss': loss.item() * config.gradient_accumulation_steps})
    
    return total_loss / len(dataloader)

# Validation function
def validate(model, dataloader, tokenizer, epoch, split_name="Val"):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} [{split_name}]")
    
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Generate predictions
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=config.max_length
            )
            
            # Prepare labels for metric calculation
            labels_for_metrics = labels.clone()
            labels_for_metrics[labels_for_metrics == -100] = tokenizer.pad_token_id
            
            all_predictions.extend(generated.cpu().numpy())
            all_labels.extend(labels_for_metrics.cpu().numpy())
            
            progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    accuracy, pred_strs, label_strs = calculate_metrics(
        all_predictions, all_labels, tokenizer
    )
    
    return avg_loss, accuracy, pred_strs, label_strs

# Plot training curves
def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history['val_accuracy'], label='Val Accuracy', marker='s', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Create confusion matrix for sample predictions
def create_analysis_plots(pred_strs, label_strs, save_prefix):
    # Character-level accuracy analysis
    char_correct = []
    for pred, label in zip(pred_strs, label_strs):
        if len(pred) == len(label):
            correct = sum([1 for p, l in zip(pred, label) if p == l])
            char_correct.append(correct / len(label) if len(label) > 0 else 0)
        else:
            char_correct.append(0)
    
    # Plot character accuracy distribution
    plt.figure(figsize=(10, 6))
    plt.hist(char_correct, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Character-level Accuracy')
    plt.ylabel('Frequency')
    plt.title('Distribution of Character-level Accuracy')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_prefix}_char_accuracy_dist.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Sample predictions table
    sample_size = min(20, len(pred_strs))
    samples_df = pd.DataFrame({
        'Input': label_strs[:sample_size],
        'Prediction': pred_strs[:sample_size],
        'Match': ['✓' if p.strip() == l.strip() else '✗' 
                  for p, l in zip(pred_strs[:sample_size], label_strs[:sample_size])]
    })
    samples_df.to_csv(f"{save_prefix}_sample_predictions.csv", index=False)

# Main training loop
def main():
    print("="*50)
    print("ByT5-small Training Pipeline")
    print("="*50)
    
    # Load tokenizer and model
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = T5ForConditionalGeneration.from_pretrained(config.model_name)
    model.to(device)
    
    print(f"Model loaded: {config.model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = WordSplitDataset('train.csv', tokenizer, config.max_length)
    val_dataset = WordSplitDataset('val.csv', tokenizer, config.max_length)
    test_dataset = WordSplitDataset('test.csv', tokenizer, config.max_length)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'epoch': []
    }
    
    best_val_accuracy = 0
    best_epoch = 0
    
    print("\n" + "="*50)
    print("Starting Training...")
    print("="*50)
    
    # Training loop
    for epoch in range(config.num_epochs):
        print(f"\n--- Epoch {epoch+1}/{config.num_epochs} ---")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, epoch)
        
        # Validate
        val_loss, val_accuracy, val_pred_strs, val_label_strs = validate(
            model, val_loader, tokenizer, epoch, "Val"
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['epoch'].append(epoch + 1)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.4f}")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            print(f"  ✓ New best model! Saving...")
            model.save_pretrained(config.save_dir)
            tokenizer.save_pretrained(config.save_dir)
            
            # Save best epoch info
            with open(f"{config.save_dir}/best_model_info.json", 'w') as f:
                json.dump({
                    'epoch': best_epoch,
                    'val_accuracy': best_val_accuracy,
                    'val_loss': val_loss,
                    'train_loss': train_loss
                }, f, indent=4)
        
        # Plot training curves (live update)
        plot_training_curves(history, f"{config.results_dir}/training_curves.png")
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Best model from epoch {best_epoch} with accuracy: {best_val_accuracy:.4f}")
    
    # Load best model for final evaluation
    print("\nLoading best model for final evaluation...")
    model = T5ForConditionalGeneration.from_pretrained(config.save_dir)
    model.to(device)
    
    # Final evaluation on all splits
    print("\n" + "="*50)
    print("Final Evaluation on All Splits")
    print("="*50)
    
    print("\n--- Training Set ---")
    train_loss, train_accuracy, train_pred_strs, train_label_strs = validate(
        model, train_loader, tokenizer, config.num_epochs, "Train"
    )
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    
    print("\n--- Validation Set ---")
    val_loss, val_accuracy, val_pred_strs, val_label_strs = validate(
        model, val_loader, tokenizer, config.num_epochs, "Val"
    )
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Accuracy: {val_accuracy:.4f}")
    
    print("\n--- Test Set ---")
    test_loss, test_accuracy, test_pred_strs, test_label_strs = validate(
        model, test_loader, tokenizer, config.num_epochs, "Test"
    )
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Create analysis plots
    print("\nGenerating analysis plots...")
    create_analysis_plots(train_pred_strs, train_label_strs, f"{config.results_dir}/train")
    create_analysis_plots(val_pred_strs, val_label_strs, f"{config.results_dir}/val")
    create_analysis_plots(test_pred_strs, test_label_strs, f"{config.results_dir}/test")
    
    # Save final results
    final_results = {
        'best_epoch': best_epoch,
        'training_epochs': config.num_epochs,
        'final_metrics': {
            'train': {'loss': train_loss, 'accuracy': train_accuracy},
            'val': {'loss': val_loss, 'accuracy': val_accuracy},
            'test': {'loss': test_loss, 'accuracy': test_accuracy}
        },
        'config': {
            'model_name': config.model_name,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'max_length': config.max_length
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(f"{config.results_dir}/final_results.json", 'w') as f:
        json.dump(final_results, f, indent=4)
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(f"{config.results_dir}/training_history.csv", index=False)
    
    print("\n" + "="*50)
    print("All results saved!")
    print(f"  - Best model: {config.save_dir}/")
    print(f"  - Training curves: {config.results_dir}/training_curves.png")
    print(f"  - Analysis plots: {config.results_dir}/")
    print(f"  - Final results: {config.results_dir}/final_results.json")
    print("="*50)

if __name__ == "__main__":
    main()