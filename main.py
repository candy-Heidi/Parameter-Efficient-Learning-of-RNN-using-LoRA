import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
import tracemalloc

def load_data_from_json(filename):
    with open(filename, "r") as f:
        dataset_dict = json.load(f)
    return dataset_dict

def count_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters count: {trainable_params}")
    return trainable_params

# Prepare DataLoader
def prepare_dataloader(input_ids, labels, batch_size=16, shuffle=True):
    dataset = torch.utils.data.TensorDataset(input_ids, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# Load and preprocess datasets
train_data_40 = load_data_from_json("train_data_40.json")
validation_data = load_data_from_json("validation_data.json")
test_data = load_data_from_json("test_data.json")

def preprocess_data(data_dict):
    input_ids = torch.tensor(data_dict["input_ids"])  # Convert input_ids to tensor
    labels = torch.tensor(data_dict["labels"])  # Convert labels to tensor
    return input_ids, labels

train_input_ids, train_labels = preprocess_data(train_data_40)
validation_input_ids, validation_labels = preprocess_data(validation_data)
test_input_ids, test_labels = preprocess_data(test_data)

# Sanitize labels to fit within vocab size
def sanitize_labels(labels, vocab_size):
    return torch.clamp(labels, max=vocab_size - 1)

train_labels = sanitize_labels(train_labels, vocab_size=50265)
validation_labels = sanitize_labels(validation_labels, vocab_size=50265)
test_labels = sanitize_labels(test_labels, vocab_size=50265)

# Create DataLoaders
train_dataloader_40 = prepare_dataloader(train_input_ids, train_labels, batch_size=16)
validation_dataloader = prepare_dataloader(validation_input_ids, validation_labels, batch_size=16, shuffle=False)
test_dataloader = prepare_dataloader(test_input_ids, test_labels, batch_size=16, shuffle=False)

# LoRA Layer definition
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / rank)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scale = alpha / rank

    def forward(self, x):
        # Assumes input x is of shape (batch_size, in_features)
        out = torch.mm(x, self.lora_A.T)  # (batch_size, rank)
        out = torch.mm(out, self.lora_B.T)  # (batch_size, out_features)
        out = out * self.scale
        return out

# LoRA RNN Cell definition (LoRA applied to hidden-to-hidden connections)
class LoRARNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, rank=4, alpha=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight_ih = nn.Linear(input_size, hidden_size)
        self.weight_hh = nn.Linear(hidden_size, hidden_size)
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.activation = nn.Tanh()
        self.lora_hh = LoRALayer(hidden_size, hidden_size, rank, alpha)

    def forward(self, input, hidden):
        ih = self.weight_ih(input)
        hh = self.weight_hh(hidden) + self.lora_hh(hidden)
        next_hidden = self.activation(ih + hh + self.bias)
        return next_hidden

# LoRA RNN model
class LoRARNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, rank=4, alpha=1, num_classes=2):
        super().__init__()
        self.cells = nn.ModuleList([LoRARNNCell(
            input_size if i == 0 else hidden_size,
            hidden_size,
            rank,
            alpha
        ) for i in range(num_layers)])
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_seq, hidden=None):
        if hidden is None:
            hidden = torch.zeros(len(self.cells), input_seq.size(1), self.cells[0].hidden_size, device=input_seq.device)
        
        output_seq = []
        for t in range(input_seq.size(0)):
            x = input_seq[t]
            for i, cell in enumerate(self.cells):
                hidden[i] = cell(x.clone(), hidden[i].clone())
                x = hidden[i]
            output_seq.append(x)
        
        output_seq = torch.stack(output_seq)
        output_seq = self.fc(output_seq)
        return output_seq, hidden

# Function to freeze RNN parameters and unfreeze LoRA parameters
def freeze_rnn_params_and_unfreeze_lora(model):
    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False  # Freeze RNN parameters
        else:
            param.requires_grad = True  # Unfreeze LoRA parameters

    # Print parameter status (for debugging)
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")

# Initialize model
input_size = 128
hidden_size = 2048
num_layers = 5
vocab_size = 50265
rank = 4
alpha = 1

lora_rnn = LoRARNN(input_size, hidden_size, num_layers, rank, alpha, num_classes=vocab_size)
lora_rnn.load_state_dict(torch.load("rnn_model.pth"), strict=False)

# Freeze RNN parameters and only train LoRA
freeze_rnn_params_and_unfreeze_lora(lora_rnn)

# Move model to GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lora_rnn.to(device)

# Function to evaluate model
def evaluate_model(model, dataloader, criterion, device=device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device).float() 
            labels = batch[1].to(device).long()
            outputs, _ = model(inputs)
            outputs = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

def count_lora_parameters(model):
    total_params = 0
    lora_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:  # 학습 가능한 파라미터만 포함
            total_params += param.numel()
            if "lora" in name:
                lora_params += param.numel()
    print(f"Total trainable parameters: {total_params}")
    print(f"LoRA parameters: {lora_params}")
    return total_params, lora_params

# Function to train LoRA RNN
def train_rnn(model, dataloader, criterion, optimizer, num_epochs=3, device='cpu'):
    model.to(device)
    model.train()

    start_time = time.time()
    tracemalloc.start()
    
    for epoch in range(num_epochs):
        total_batches = len(dataloader)
        total_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch[0].to(device).float() 
            labels = batch[1].to(device).long()
            
            optimizer.zero_grad()

            outputs, _ = model(inputs)
            outputs = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            # Print training progress
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(dataloader)
        perplexity = torch.exp(torch.tensor(avg_loss))
        print(f'Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}')
    
    # Print training time
    end_time = time.time()
    training_time = end_time - start_time
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gpu_memory_allocated = torch.cuda.memory_allocated(device) / 10**6
    gpu_memory_peak = torch.cuda.max_memory_allocated(device) / 10**6

    print(f"Training Time (LoRA): {training_time:.2f} seconds")
    print(f"Current CPU Memory Usage: {current / 10**6:.2f} MB")
    print(f"Peak CPU Memory Usage: {peak / 10**6:.2f} MB")
    print(f"Current GPU Memory Usage: {gpu_memory_allocated:.2f} MB")
    print(f"Peak GPU Memory Usage: {gpu_memory_peak:.2f} MB")
    
    return training_time, current, peak, gpu_memory_allocated, gpu_memory_peak

# Optimizer (only for LoRA parameters)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, lora_rnn.parameters()), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train and evaluate model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_rnn(lora_rnn, train_dataloader_40, criterion, optimizer, num_epochs=1, device=device)
total_params, lora_params = count_lora_parameters(lora_rnn)


# Validation evaluation
val_loss, val_accuracy = evaluate_model(lora_rnn, validation_dataloader, criterion, device=device)
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Test evaluation
test_loss, test_accuracy = evaluate_model(lora_rnn, test_dataloader, criterion, device=device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
