import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import jiwer
from tqdm import tqdm
import gc
import os

# ---------------------------
# Environment / determinism
# ---------------------------
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Free memory if possible
if torch.cuda.is_available():
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
gc.collect()

# ---------------------------
# Load CSV files
# ---------------------------
print("Loading data...")
train_df = pd.read_csv('/DATA/rohit/NLP_2025/dataset/pretok/hindi/hindi_train.csv')
val_df = pd.read_csv('/DATA/rohit/NLP_2025/dataset/pretok/hindi/hindi_val.csv')
test_df = pd.read_csv('/DATA/rohit/NLP_2025/dataset/pretok/hindi/hindi_test.csv')

print(f"Train samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")

print(f"\nTrain columns: {train_df.columns.tolist()}")
print(f"Validation columns: {val_df.columns.tolist()}")
print(f"Test columns: {test_df.columns.tolist()}")

# Automatic column detection
def get_column_names(df):
    cols = df.columns.tolist()
    input_col = None
    output_col = None

    for col in cols:
        col_lower = col.lower().strip()
        if col_lower in ['word', 'input', 'source', 'text', 'sandhi']:
            input_col = col
            break

    for col in cols:
        col_lower = col.lower().strip()
        if col_lower in ['split', 'output', 'target', 'label', 'unsandhi']:
            output_col = col
            break

    if input_col is None:
        input_col = cols[0]
    if output_col is None:
        output_col = cols[1] if len(cols) > 1 else cols[0]

    return input_col, output_col

input_col_train, output_col_train = get_column_names(train_df)
input_col_val, output_col_val = get_column_names(val_df)
input_col_test, output_col_test = get_column_names(test_df)

print(f"\nUsing columns:")
print(f"Train: input='{input_col_train}', output='{output_col_train}'")
print(f"Validation: input='{input_col_val}', output='{output_col_val}'")
print(f"Test: input='{input_col_test}', output='{output_col_test}'")

# Rename for consistency
train_df = train_df.rename(columns={input_col_train: 'word', output_col_train: 'split'})
val_df   = val_df.rename(columns={input_col_val: 'word',   output_col_val: 'split'})
test_df  = test_df.rename(columns={input_col_test: 'word',  output_col_test: 'split'})

print("\nFirst 3 training samples:")
print(train_df[['word', 'split']].head(3))

# ---------------------------
# Build char vocabs
# ---------------------------
def build_vocab(texts):
    chars = set()
    for text in texts:
        chars.update(str(text))
    return chars

all_words = pd.concat([train_df['word'], val_df['word'], test_df['word']]).astype(str)
all_splits = pd.concat([train_df['split'], val_df['split'], test_df['split']]).astype(str)

input_chars = sorted(list(build_vocab(all_words)))
output_chars = sorted(list(build_vocab(all_splits)))

input_chars = ['<PAD>', '<UNK>'] + input_chars
output_chars = ['<PAD>', '<START>', '<END>', '<UNK>'] + output_chars

input_char_to_idx = {char: idx for idx, char in enumerate(input_chars)}
output_char_to_idx = {char: idx for idx, char in enumerate(output_chars)}
idx_to_output_char = {idx: char for char, idx in output_char_to_idx.items()}

input_vocab_size = len(input_chars)
output_vocab_size = len(output_chars)

print(f"Input vocabulary size: {input_vocab_size}")
print(f"Output vocabulary size: {output_vocab_size}")

# Global max lengths (still useful as an upper bound)
max_input_len = max(all_words.apply(lambda x: len(str(x))))
max_output_len = max(all_splits.apply(lambda x: len(str(x)))) + 2  # start & end

print(f"Max input length: {max_input_len}")
print(f"Max output length: {max_output_len}")

# ---------------------------
# Dataset (keeps padded tensors but collate will trim per-batch)
# ---------------------------
class SandhiDataset(Dataset):
    def __init__(self, df, input_char_to_idx, output_char_to_idx, max_input_len, max_output_len):
        self.df = df.reset_index(drop=True)
        self.input_char_to_idx = input_char_to_idx
        self.output_char_to_idx = output_char_to_idx
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        
    def __len__(self):
        return len(self.df)
    
    def encode_input(self, text):
        text = str(text)
        encoded = [self.input_char_to_idx.get(char, self.input_char_to_idx['<UNK>']) for char in text]
        encoded = encoded[:self.max_input_len]
        encoded += [self.input_char_to_idx['<PAD>']] * (self.max_input_len - len(encoded))
        return torch.LongTensor(encoded)
    
    def encode_output(self, text):
        text = str(text)
        encoded = [self.output_char_to_idx['<START>']]
        encoded += [self.output_char_to_idx.get(char, self.output_char_to_idx['<UNK>']) for char in text]
        encoded.append(self.output_char_to_idx['<END>'])
        encoded = encoded[:self.max_output_len]
        encoded += [self.output_char_to_idx['<PAD>']] * (self.max_output_len - len(encoded))
        return torch.LongTensor(encoded)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        encoder_input = self.encode_input(row['word'])
        decoder_output = self.encode_output(row['split'])
        decoder_input = decoder_output[:-1]  # remove last
        decoder_target = decoder_output[1:]  # remove first
        return encoder_input, decoder_input, decoder_target

# Create datasets
train_dataset = SandhiDataset(train_df, input_char_to_idx, output_char_to_idx, max_input_len, max_output_len)
val_dataset   = SandhiDataset(val_df, input_char_to_idx, output_char_to_idx, max_input_len, max_output_len)
test_dataset  = SandhiDataset(test_df, input_char_to_idx, output_char_to_idx, max_input_len, max_output_len)

# ---------------------------
# Collate function: stack but trim per-batch to save memory
# ---------------------------
pad_enc = input_char_to_idx['<PAD>']
pad_dec = output_char_to_idx['<PAD>']

def collate_trim(batch):
    # batch: list of tuples (enc_in, dec_in, dec_tgt) each already padded to global max
    encs = torch.stack([item[0] for item in batch], dim=0)  # (B, max_input_len)
    dec_ins = torch.stack([item[1] for item in batch], dim=0)  # (B, max_dec_len)
    dec_tgts = torch.stack([item[2] for item in batch], dim=0)

    # Trim encoder to actual max non-pad length in this batch
    enc_mask = encs != pad_enc  # (B, L)
    if enc_mask.any():
        enc_max_len = int(enc_mask.sum(dim=1).max().item())
        encs = encs[:, :enc_max_len]
    else:
        encs = encs[:, :1]

    # Trim decoder to actual max non-pad length in this batch
    dec_mask = dec_ins != pad_dec
    if dec_mask.any():
        dec_max_len = int(dec_mask.sum(dim=1).max().item())
        dec_ins = dec_ins[:, :dec_max_len]
        dec_tgts = dec_tgts[:, :dec_max_len]
    else:
        dec_ins = dec_ins[:, :1]
        dec_tgts = dec_tgts[:, :1]

    return encs, dec_ins, dec_tgts

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_trim)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_trim)
test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_trim)

# ---------------------------
# Vectorized Bahdanau Attention
# ---------------------------
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.V  = nn.Linear(hidden_size, 1, bias=True)
    
    def forward(self, query, values):
        """
        query: (batch, seq_q, hidden)
        values: (batch, seq_v, hidden)
        returns:
            context: (batch, seq_q, hidden)
            attention_weights: (batch, seq_q, seq_v)
        """
        # Apply linear layers
        # W1(query) -> (batch, seq_q, hidden)
        # W2(values) -> (batch, seq_v, hidden)
        W1_q = self.W1(query)       # (B, Q, H)
        W2_v = self.W2(values)      # (B, V, H)

        # Broadcast sum: (B, Q, 1, H) + (B, 1, V, H) -> (B, Q, V, H)
        score_tanh = torch.tanh(W1_q.unsqueeze(2) + W2_v.unsqueeze(1))  # (B, Q, V, H)

        # Apply V: linear on last dim -> (B, Q, V, 1) -> squeeze -> (B, Q, V)
        scores = self.V(score_tanh).squeeze(-1)

        # attention weights over encoder sequence (seq_v) for every query position
        attention_weights = torch.softmax(scores, dim=2)  # (B, Q, V)

        # Weighted sum: (B, Q, V) x (B, V, H) -> (B, Q, H)
        context = torch.matmul(attention_weights, values)

        return context, attention_weights

# ---------------------------
# BiLSTM Encoder
# ---------------------------
class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(BiLSTMEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.hidden_size = hidden_size

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.bilstm(embedded)
        # hidden: (2, batch, hidden) -> concat forward & backward
        hidden_cat = torch.cat([hidden[0], hidden[1]], dim=1)  # (batch, hidden*2)
        cell_cat = torch.cat([cell[0], cell[1]], dim=1)
        return outputs, hidden_cat, cell_cat

# ---------------------------
# BiLSTM Decoder using vectorized attention
# ---------------------------
class BiLSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(BiLSTMDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.attention = BahdanauAttention(hidden_size * 2)  # because bilstm outputs hidden*2
        self.fc = nn.Linear(hidden_size * 4, hidden_size * 2)
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(hidden_size * 2, vocab_size)
        self.hidden_size = hidden_size

    def forward(self, x, encoder_outputs, hidden_cat, cell_cat):
        """
        x: (batch, dec_seq_len)
        encoder_outputs: (batch, enc_seq_len, hidden*2)
        hidden_cat: (batch, hidden*2)
        cell_cat: (batch, hidden*2)
        """
        embedded = self.embedding(x)  # (B, dec_seq_len, emb)
        
        # Re-create initial hidden and cell for bidirectional LSTM from concatenated vectors
        h_f = hidden_cat[:, :self.hidden_size]
        h_b = hidden_cat[:, self.hidden_size:]
        c_f = cell_cat[:, :self.hidden_size]
        c_b = cell_cat[:, self.hidden_size:]
        hidden_init = torch.stack([h_f, h_b], dim=0)  # (2, B, H)
        cell_init = torch.stack([c_f, c_b], dim=0)

        decoder_outputs, (hidden_new, cell_new) = self.bilstm(embedded, (hidden_init, cell_init))
        # decoder_outputs: (B, dec_seq_len, hidden*2)

        # Vectorized attention over all decoder time steps
        # Query = decoder_outputs (B, Q, H*2), values = encoder_outputs (B, V, H*2)
        context_vectors, attn_w = self.attention(decoder_outputs, encoder_outputs)  # (B, Q, H*2)

        # Combine and predict
        combined = torch.cat([decoder_outputs, context_vectors], dim=2)  # (B, Q, H*4)
        out = self.fc(combined)  # (B, Q, H*2)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.output(out)  # (B, Q, vocab)
        
        # pack hidden back to concatenated form for any external use
        hidden_new_cat = torch.cat([hidden_new[0], hidden_new[1]], dim=1)  # (B, H*2)
        cell_new_cat   = torch.cat([cell_new[0], cell_new[1]], dim=1)

        return out, hidden_new_cat, cell_new_cat

# ---------------------------
# Seq2Seq wrapper
# ---------------------------
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        encoder_outputs, hidden, cell = self.encoder(src)
        outputs, _, _ = self.decoder(trg, encoder_outputs, hidden, cell)
        return outputs

# ---------------------------
# Build model (small dims preserved)
# ---------------------------
embedding_dim = 32
hidden_size = 64

encoder = BiLSTMEncoder(input_vocab_size, embedding_dim, hidden_size)
decoder = BiLSTMDecoder(output_vocab_size, embedding_dim, hidden_size)
model = Seq2Seq(encoder, decoder).to(device)

print("\nModel Architecture:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

if torch.cuda.is_available():
    try:
        print(f"GPU Memory after model: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB allocated")
        print(f"GPU Memory after model: {torch.cuda.memory_reserved(0)/1024**3:.2f} GB reserved")
    except Exception:
        pass

# ---------------------------
# Loss / optimizer / scheduler
# ---------------------------
criterion = nn.CrossEntropyLoss(ignore_index=output_char_to_idx['<PAD>'])
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# AMP scaler
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

# ---------------------------
# Training / validation loops
# ---------------------------
def train_epoch(model, dataloader, criterion, optimizer, device, accumulation_steps=4):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")
    
    for idx, (encoder_input, decoder_input, decoder_target) in pbar:
        # Move to device
        encoder_input = encoder_input.to(device)
        decoder_input = decoder_input.to(device)
        decoder_target = decoder_target.to(device)

        # Mixed precision context
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(encoder_input, decoder_input)  # (B, dec_len, vocab)
            outputs_flat = outputs.reshape(-1, outputs.shape[-1])  # (B*dec_len, vocab)
            decoder_target_flat = decoder_target.reshape(-1)       # (B*dec_len)
            loss = criterion(outputs_flat, decoder_target_flat)
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

        # Accuracy calculation (without grad)
        with torch.no_grad():
            preds = outputs_flat.argmax(dim=1)
            mask = decoder_target_flat != output_char_to_idx['<PAD>']
            correct += ((preds == decoder_target_flat) & mask).sum().item()
            total += mask.sum().item()

        # periodic cleanup
        if idx % 10 == 0:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()

        pbar.set_postfix({'loss': f"{(total_loss/(idx+1)):.4f}", 'acc': f"{(correct/total if total>0 else 0):.4f}"})

    # leftover gradients
    if (idx + 1) % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, (encoder_input, decoder_input, decoder_target) in enumerate(tqdm(dataloader, desc="Validating")):
            encoder_input = encoder_input.to(device)
            decoder_input = decoder_input.to(device)
            decoder_target = decoder_target.to(device)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(encoder_input, decoder_input)
                outputs_flat = outputs.reshape(-1, outputs.shape[-1])
                decoder_target_flat = decoder_target.reshape(-1)
                loss = criterion(outputs_flat, decoder_target_flat)
                total_loss += loss.item()

                preds = outputs_flat.argmax(dim=1)
                mask = decoder_target_flat != output_char_to_idx['<PAD>']
                correct += ((preds == decoder_target_flat) & mask).sum().item()
                total += mask.sum().item()

            if idx % 10 == 0:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                gc.collect()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy

# ---------------------------
# Training loop with Early Stopping
# ---------------------------
print("\nTraining model...")

num_epochs = 50          # can increase safely due to early stopping
patience = 8             # stop after 8 epochs with no val_loss improvement
best_val_loss = float('inf')
patience_counter = 0

# to store training history
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

best_model_wts = None

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    # record stats
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    # report
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

    # learning rate scheduler step
    scheduler.step(val_loss)
    
    # check for improvement
    if val_loss < best_val_loss - 1e-4:  # small threshold to prevent floating noise
        best_val_loss = val_loss
        patience_counter = 0
        best_model_wts = model.state_dict()  # save weights in memory
        torch.save(best_model_wts, 'best_model.pt')
        print(f"✅ Validation loss improved → model saved (Val Loss: {best_val_loss:.4f})")
    else:
        patience_counter += 1
        print(f"⏳ No improvement for {patience_counter} epoch(s).")

        if patience_counter >= patience:
            print(f"\n⛔ Early stopping triggered at epoch {epoch+1}!")
            break

# restore best weights after training
if best_model_wts is not None:
    model.load_state_dict(best_model_wts)
    print("\n✅ Loaded best model weights from memory.")
elif os.path.exists('best_model.pt'):
    model.load_state_dict(torch.load('best_model.pt', map_location=device))
    print("\n✅ Loaded best_model.pt from disk.")
else:
    print("\n⚠️ No best model found; continuing with current weights.")

# ---------------------------
# Load best model (if exists)
# ---------------------------
if os.path.exists('best_model.pt'):
    model.load_state_dict(torch.load('best_model.pt', map_location=device))
    print("Loaded best_model.pt")
else:
    print("No saved model found; continuing with current weights.")

# ---------------------------
# Prediction (greedy) using decoder iteratively
# ---------------------------
def predict(model, input_text, input_char_to_idx, idx_to_output_char, output_char_to_idx, max_input_len, max_output_len, device):
    model.eval()
    with torch.no_grad():
        # Encode input
        encoded = [input_char_to_idx.get(char, input_char_to_idx['<UNK>']) for char in str(input_text)]
        encoded = encoded[:max_input_len]
        encoded += [input_char_to_idx['<PAD>']] * (max_input_len - len(encoded))
        encoder_input = torch.LongTensor(encoded).unsqueeze(0).to(device)

        # Encoder output
        encoder_outputs, hidden, cell = model.encoder(encoder_input)

        # Start decoding with <START>
        decoder_input = torch.LongTensor([[output_char_to_idx['<START>']]]).to(device)
        decoded_chars = []

        for _ in range(max_output_len - 2):
            output, hidden, cell = model.decoder(decoder_input, encoder_outputs, hidden, cell)
            # Take last time step’s prediction
            predicted_id = output[:, -1, :].argmax(dim=1).item()
            predicted_char = idx_to_output_char[predicted_id]

            # Stop at <END> or <PAD>
            if predicted_char in ['<END>', '<PAD>']:
                break
            if predicted_char not in ['<START>', '<UNK>']:
                decoded_chars.append(predicted_char)

            # The *only* next input token is the one just predicted
            decoder_input = torch.LongTensor([[predicted_id]]).to(device)

        return ''.join(decoded_chars)


# ---------------------------
# Evaluate on test set
# ---------------------------
print("\nEvaluating on test set...")
predictions = []
references = []

for i in range(len(test_df)):
    input_text = test_df['word'].iloc[i]
    true_output = str(test_df['split'].iloc[i])

    pred = predict(model, input_text, input_char_to_idx, idx_to_output_char, 
                   output_char_to_idx, max_input_len, max_output_len, device)

    predictions.append(pred)
    references.append(true_output)

    if i < 10:
        print(f"Input: {input_text}")
        print(f"True:  {true_output}")
        print(f"Pred:  {pred}")
        print()

# ---------------------------
# Metrics
# ---------------------------
wer = jiwer.wer(references, predictions)
cer = jiwer.cer(references, predictions)

exact_matches = sum([1 for p, r in zip(predictions, references) if p == r])
accuracy = exact_matches / len(predictions) if len(predictions) > 0 else 0.0

total_chars = sum([len(r) for r in references])
correct_chars = 0
for pred, ref in zip(predictions, references):
    min_len = min(len(pred), len(ref))
    for j in range(min_len):
        if pred[j] == ref[j]:
            correct_chars += 1

char_accuracy = correct_chars / total_chars if total_chars > 0 else 0.0

print(f"\nWord Error Rate (WER): {wer:.4f}")
print(f"Character Error Rate (CER): {cer:.4f}")
print(f"Exact Match Accuracy: {accuracy:.4f} ({exact_matches}/{len(predictions) if len(predictions)>0 else 0})")
print(f"Character-level Accuracy: {char_accuracy:.4f}")

# ---------------------------
# Plot training history (if any)
# ---------------------------
if len(history['train_loss']) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss\n(PyTorch BiLSTM Encoder-Decoder)', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy\n(PyTorch BiLSTM Encoder-Decoder)', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history_pytorch.png', dpi=300, bbox_inches='tight')
    plt.show()

# Simple test metrics bar plot
fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['WER', 'CER', 'Exact Match\nAccuracy', 'Character\nAccuracy']
values = [wer, cer, accuracy, char_accuracy]
# choose colors programmatically: red for high error rates else green
colors = ['#e74c3c' if v > 0.5 else '#2ecc71' for v in [wer, cer]] + ['#2ecc71', '#2ecc71']

bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Test Set Evaluation Metrics\nPyTorch BiLSTM Encoder-Decoder with Bahdanau Attention', fontsize=14, fontweight='bold', pad=20)
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#e74c3c', alpha=0.7, label='Error Rate (lower is better)'),
    Patch(facecolor='#2ecc71', alpha=0.7, label='Accuracy (higher is better)')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('test_metrics_pytorch.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print(" FINAL TEST RESULTS - PyTorch BiLSTM ENCODER-DECODER")
print("="*70)
print(f"Word Error Rate (WER):       {wer:.4f}")
print(f"Character Error Rate (CER):  {cer:.4f}")
print(f"Exact Match Accuracy:        {accuracy:.4f} ({exact_matches}/{len(predictions)})")
print(f"Character-level Accuracy:    {char_accuracy:.4f}")
print("="*70)
print("\nModel Architecture:")
print("- Encoder: Bidirectional LSTM")
print("- Decoder: Bidirectional LSTM")
print("- Attention: Vectorized Bahdanau Attention Mechanism")
print("- Framework: PyTorch")
print("="*70)
