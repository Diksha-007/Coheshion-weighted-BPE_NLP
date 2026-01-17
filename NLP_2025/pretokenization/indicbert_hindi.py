import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer  # ✅ replaced
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
import unicodedata

# ------------------------- #
# 1️⃣ Model Components
# ------------------------- #
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("ai4bharat/IndicBERTv2-MLM-Sam-TLM")  # ✅ IndicBERT v2

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

# (BahdanauAttention, Decoder, SandhiSeq2Seq classes remain the same)
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, hidden, encoder_outputs):
        hidden = hidden.permute(1, 0, 2)
        score = self.V(torch.tanh(self.W1(encoder_outputs) + self.W2(hidden)))
        attn_weights = torch.softmax(score, dim=1)
        context = torch.sum(attn_weights * encoder_outputs, dim=1)
        return context, attn_weights

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim=768, embedding_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.GRU(embedding_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.attention = BahdanauAttention(hidden_dim)

    def forward(self, input_token, hidden, encoder_outputs):
        embedded = self.embedding(input_token).unsqueeze(1)
        context, attn_weights = self.attention(hidden, encoder_outputs)
        rnn_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)
        #output, hidden = self.rnn(rnn_input, hidden)
        output, hidden = self.rnn(rnn_input, hidden.contiguous())
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, attn_weights

class SandhiSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, input_enc, target_seq, teacher_forcing_ratio=0.5):
        src_input_ids = input_enc["input_ids"].squeeze(1).to(self.device)
        attention_mask = input_enc["attention_mask"].squeeze(1).to(self.device)
        trg_input_ids = target_seq.to(self.device)

        encoder_outputs = self.encoder(src_input_ids, attention_mask)
        hidden = encoder_outputs[:, -1, :].unsqueeze(0)

        batch_size = src_input_ids.size(0)
        trg_len = trg_input_ids.size(1)
        output_dim = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, trg_len, output_dim).to(self.device)

        input_token = trg_input_ids[:, 0]

        for t in range(1, trg_len):
            output, hidden, _ = self.decoder(input_token, hidden, encoder_outputs)
            

            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = trg_input_ids[:, t] if teacher_force else top1

        return outputs


# ------------------------- #
# 2️⃣ Data Preparation
# ------------------------- #
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBERTv2-MLM-Sam-TLM")  # ✅

train_df = pd.read_csv("hindi_train.csv")
val_df   = pd.read_csv("hindi_val.csv")
test_df  = pd.read_csv("hindi_test.csv")

# (vocab creation + encode_example remain same)


# Build character vocabulary
all_text = list(train_df["Word"].astype(str)) + list(train_df["Split"].astype(str))
all_chars = set("".join(all_text))
special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
vocab = special_tokens + sorted(list(all_chars))
char2idx = {ch: idx for idx, ch in enumerate(vocab)}
idx2char = {idx: ch for ch, idx in char2idx.items()}
pad_idx = char2idx["<PAD>"]

def encode_example(word, split):
    input_enc = tokenizer(word, truncation=True, padding='max_length', max_length=16, return_tensors='pt')
    target_seq = [char2idx["<SOS>"]] + [char2idx.get(ch, char2idx["<UNK>"]) for ch in split] + [char2idx["<EOS>"]]
    target_seq = target_seq[:32]
    target_seq += [pad_idx] * (32 - len(target_seq))
    return input_enc, torch.tensor(target_seq)

class SandhiDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        word = str(self.df.iloc[idx]["Word"])
        split = str(self.df.iloc[idx]["Split"])
        input_enc, target_seq = encode_example(word, split)
        return input_enc, target_seq

train_loader = DataLoader(SandhiDataset(train_df), batch_size=8, shuffle=True)
val_loader   = DataLoader(SandhiDataset(val_df), batch_size=8)



# ------------------------- #
# 3️⃣ Model Training (Early Stopping + LR Scheduler)
# ------------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

encoder = Encoder()
decoder = Decoder(output_dim=len(vocab), hidden_dim=768, embedding_dim=256)
model = SandhiSeq2Seq(encoder, decoder, device).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# 🔁 Reduce learning rate when validation loss plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-7
)

train_losses = []
val_losses = []

best_val_loss = float('inf')
patience = 4      # Stop if no improvement for these many epochs
counter = 0         # Tracks epochs without improvement

for epoch in range(30):   # You can keep a high max epoch
    model.train()
    train_loss = 0

    # ------------------------- #
    # Training Loop
    # ------------------------- #
    for input_enc, target_seq in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        optimizer.zero_grad()
        target_seq = target_seq.to(device)
        outputs = model(input_enc, target_seq)
        output_dim = outputs.shape[-1]
        outputs = outputs[:, 1:].reshape(-1, output_dim)
        target = target_seq[:, 1:].reshape(-1)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ------------------------- #
    # Validation Loop
    # ------------------------- #
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for input_enc, target_seq in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
            target_seq = target_seq.to(device)
            outputs = model(input_enc, target_seq, teacher_forcing_ratio=0)
            output_dim = outputs.shape[-1]
            outputs = outputs[:, 1:].reshape(-1, output_dim)
            target = target_seq[:, 1:].reshape(-1)
            loss = criterion(outputs, target)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # ------------------------- #
    # Logging + LR Scheduling
    # ------------------------- #
    print(f"\nEpoch {epoch+1} — Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    scheduler.step(avg_val_loss)  # 🔁 Adjust LR based on validation loss
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Current Learning Rate: {current_lr:.6e}")

    # ------------------------- #
    # Early Stopping Check
    # ------------------------- #
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        torch.save(model.state_dict(), "best_sandhi_model_hindi.pt")
        print("✅ Saved new best model!")
    else:
        counter += 1
        print(f"⚠️ No improvement for {counter} epoch(s).")
        if counter >= patience:
            print(f"⏹️ Early stopping triggered at epoch {epoch+1}.")
            break

    # ------------------------- #
    # Print Sample Predictions
    # ------------------------- #
    model.eval()
    print("\nSample Predictions:")
    for word in val_df["Word"][:3]:
        pred = []
        with torch.no_grad():
            enc = tokenizer(word, return_tensors="pt").to(device)
            enc_out = model.encoder(enc["input_ids"].squeeze(1), enc["attention_mask"].squeeze(1))
            hidden = enc_out[:, -1, :].unsqueeze(0)
            input_token = torch.tensor([char2idx["<SOS>"]]).to(device)
            for _ in range(30):
                out, hidden, _ = model.decoder(input_token, hidden, enc_out)
                top1 = out.argmax(1)
                if top1.item() == char2idx["<EOS>"]:
                    break
                pred.append(idx2char[top1.item()])
                input_token = top1
        print(f"  {word} → {''.join(pred)}")




# ------------------------- #
# 4️⃣ Plot Train/Val Loss
# ------------------------- #
plt.figure(figsize=(7,4))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()
plt.savefig("train_val_loss_hindi.png")
print("✅ Saved plot as train_val_loss.png")


# ------------------------- #
# 5️⃣ Evaluate Accuracy
# ------------------------- #
def predict_sandhi(model, word, max_len=30):
    model.eval()
    with torch.no_grad():
        input_enc = tokenizer(word, return_tensors="pt").to(device)
        encoder_outputs = model.encoder(input_enc["input_ids"].squeeze(1),
                                        input_enc["attention_mask"].squeeze(1))
        hidden = encoder_outputs[:, -1, :].unsqueeze(0)
        input_token = torch.tensor([char2idx["<SOS>"]]).to(device)
        outputs = []
        for _ in range(max_len):
            output, hidden, _ = model.decoder(input_token, hidden, encoder_outputs)
            top1 = output.argmax(1)
            if top1.item() == char2idx["<EOS>"]:
                break
            outputs.append(top1.item())
            input_token = top1
        return "".join(idx2char[i] for i in outputs)

# Load best model for evaluation
model.load_state_dict(torch.load("best_sandhi_model_hindi.pt"))
model.eval()

preds = []
for word in tqdm(test_df["Word"], desc="Evaluating"):
    preds.append(predict_sandhi(model, word))

test_df["Predicted_Split"] = preds
test_df["Char_Correct"] = test_df.apply(lambda r: sum(a==b for a,b in zip(r["Split"], r["Predicted_Split"])) / max(len(r["Split"]),1), axis=1)
test_df["Exact_Match"] = (test_df["Split"] == test_df["Predicted_Split"])

print(f"✅ Character-wise Accuracy: {test_df['Char_Correct'].mean()*100:.2f}%")
print(f"✅ Exact Match Accuracy: {test_df['Exact_Match'].mean()*100:.2f}%")