# rnn only
# data : iwslt2017-en-de
# 지수 
#   1- CrossEntropyLoss - 모델의 출력이 실제 레이블과 얼마나 차이가 나는지를 측정
#   2- perplexity 

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import time
import tracemalloc

# RNNCell과 RNN 클래스 정의
class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Standard RNN weights
        self.weight_ih = nn.Linear(input_size, hidden_size)
        self.weight_hh = nn.Linear(hidden_size, hidden_size)
        
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.activation = nn.Tanh()

    def forward(self, input, hidden):
        ih = self.weight_ih(input)
        hh = self.weight_hh(hidden)
        next_hidden = self.activation(ih + hh + self.bias)
        return next_hidden

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, num_classes=2):
        super().__init__() 
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.cells = nn.ModuleList([
            RNNCell(
                input_size if i == 0 else hidden_size,
                hidden_size
            ) for i in range(num_layers)
        ])
        
        # 최종 출력 레이어 설정
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_seq, hidden=None):
        if hidden is None:
            hidden = torch.zeros(self.num_layers, input_seq.size(1), self.hidden_size, device=input_seq.device)
        
        output_seq = []
        for t in range(input_seq.size(0)):
            x = input_seq[t]
            for i, cell in enumerate(self.cells):
                hidden[i] = cell(x.clone(), hidden[i].clone())
                x = hidden[i]
            output_seq.append(x)
        
        output_seq = torch.stack(output_seq)
        output_seq = self.fc(output_seq)  # Fully connected layer에서 클래스 수로 변환
        return output_seq, hidden

# 데이터셋 로드 및 전처리
dataset = load_dataset("iwslt2017", "iwslt2017-en-de",trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50")

def tokenize_function(examples):
    source = [ex["en"] for ex in examples["translation"]]
    target = [ex["de"] for ex in examples["translation"]]
    
    source_tokenized = tokenizer(source, padding="max_length", truncation=True, max_length=128)
    target_tokenized = tokenizer(target, padding="max_length", truncation=True, max_length=128)
    
    source_tokenized["labels"] = target_tokenized["input_ids"]
    return source_tokenized

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# PyTorch 데이터셋으로 변환 (input_ids와 labels 포함)
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'labels'])
train_dataset = tokenized_datasets["train"]
small_train_dataset = torch.utils.data.Subset(train_dataset, range(0, 5000))  # 5000개 샘플만 사용
train_dataloader = DataLoader(small_train_dataset, batch_size=16, shuffle=True)
print(f"Dataset size: {len(small_train_dataset)}")

# 모델 생성
input_size = 128
hidden_size = 256
num_layers = 2
vocab_size = tokenizer.vocab_size

# 순수 RNN 모델 생성
rnn = RNN(input_size, hidden_size, num_layers, num_classes=vocab_size)

# 파라미터 개수 계산
param_count = sum(p.numel() for p in rnn.parameters() if p.requires_grad)
print(f"Parameter Count: {param_count}")

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), lr=0.001)

# 학습 루프
def train_rnn(model, dataloader, criterion, optimizer, num_epochs=3, device='cpu'):
    torch.autograd.set_detect_anomaly(True)
    model.to(device)
    model.train()

    start_time = time.time()
    tracemalloc.start()
    
    for epoch in range(num_epochs):
        total_batches = len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Total batches: {total_batches}")
        total_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch['input_ids'].to(device).float() 
            labels = batch['labels'].to(device).long()
            
            optimizer.zero_grad()

            outputs, _ = model(inputs)
            outputs = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        perplexity = torch.exp(torch.tensor(avg_loss))
        print(f'Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}')
    
    # 학습 시간 및 메모리 사용량 출력
    end_time = time.time()
    training_time = end_time - start_time

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    gpu_memory_allocated = torch.cuda.memory_allocated(device) / 10**6
    gpu_memory_peak = torch.cuda.max_memory_allocated(device) / 10**6

    return training_time, current, peak, gpu_memory_allocated, gpu_memory_peak

# 모델 학습 및 결과 출력 (GPU 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
training_time, current_memory, peak_memory, gpu_memory_allocated, gpu_memory_peak = train_rnn(rnn, train_dataloader, criterion, optimizer, num_epochs=3, device=device)


print(f"Training Time: {training_time:.2f} seconds")
print(f"Current CPU Memory Usage: {current_memory / 10**6:.2f} MB")
print(f"Peak CPU Memory Usage: {peak_memory / 10**6:.2f} MB")
print(f"Current GPU Memory Usage: {gpu_memory_allocated:.2f} MB")
print(f"Peak GPU Memory Usage: {gpu_memory_peak:.2f} MB")
