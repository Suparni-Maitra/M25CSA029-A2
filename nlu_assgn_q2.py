# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# Load Data
def load_data(filename="TrainingNames.txt"):
    with open(filename, 'r', encoding='utf-8') as f:
        names = [line.strip().lower() for line in f if line.strip()]

    chars = sorted(list(set(''.join(names) + '.#')))
    char_to_int = {c: i for i, c in enumerate(chars)}
    int_to_char = {i: c for i, c in enumerate(chars)}
    return names, chars, char_to_int, int_to_char

names, vocab, char_to_int, int_to_char = load_data()
vocab_size = len(vocab)

def get_random_batch(names, char_to_int):
    name = random.choice(names)
    # Input: ".name" -> Output: "name#"
    input_seq = [char_to_int['.']] + [char_to_int[c] for c in name]
    target_seq = [char_to_int[c] for c in name] + [char_to_int['#']]
    return torch.tensor(input_seq).unsqueeze(1), torch.tensor(target_seq)

class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Parameters: W_xh (input to hidden), W_hh (hidden to hidden), W_hy (hidden to output)
        self.W_xh = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.W_hy = nn.Parameter(torch.randn(hidden_size, output_size) * 0.01)
        self.b_h = nn.Parameter(torch.zeros(1, hidden_size))
        self.b_y = nn.Parameter(torch.zeros(1, output_size))

    def forward(self, x, h):
        # h_t = tanh(x_t * W_xh + h_{t-1} * W_hh + b_h)
        h = torch.tanh(torch.matmul(x, self.W_xh) + torch.matmul(h, self.W_hh) + self.b_h)
        y = torch.matmul(h, self.W_hy) + self.b_y
        return y, h

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Combined parameters for the 4 gates (input, forget, cell, output)
        self.W_raw = nn.Parameter(torch.randn(input_size + hidden_size, hidden_size * 4) * 0.01)
        self.b_raw = nn.Parameter(torch.zeros(1, hidden_size * 4))
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h, c):
        combined = torch.cat((x, h), 1)
        gates = torch.matmul(combined, self.W_raw) + self.b_raw
        i, f, g, o = gates.chunk(4, 1)

        i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        y = self.fc(h_next)
        return y, h_next, c_next

    def init_hidden(self):
        # LSTMs require both a hidden state (h) and a cell state (c)
        return torch.zeros(1, self.hidden_size), torch.zeros(1, self.hidden_size)

# 3. RNN with Basic Attention
class AttentionRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        # The internal RNN that generates the hidden states
        self.rnn = VanillaRNN(input_size, hidden_size, hidden_size)
        # Linear layer to calculate attention scores
        self.attn_linear = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h, history):
        # 1. Get current hidden state from internal RNN
        y_rnn, h_next = self.rnn(x, h)

        if len(history) == 0:
            return self.fc(h_next), h_next

        # 2. Calculate Attention
        # Stack previous hidden states
        hist_tensor = torch.stack(history).squeeze(1) # [seq_len, hidden_size]

        # Expand current hidden state to match history length
        curr_h_expanded = h_next.repeat(len(history), 1) # [seq_len, hidden_size]

        # Combine current state with each historical state to get weights
        combined = torch.cat((hist_tensor, curr_h_expanded), 1) # [seq_len, hidden*2]
        attn_weights = torch.softmax(self.attn_linear(combined), dim=0) # [seq_len, 1]

        # Context vector is the weighted sum of history
        context = torch.sum(attn_weights * hist_tensor, dim=0, keepdim=True) # [1, hidden_size]

        return self.fc(context), h_next

    def init_hidden(self):

        return self.rnn.init_hidden()

def train_model(model, names, char_to_int, epochs=3000):
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for i in range(epochs):
        input_tensor, target_tensor = get_random_batch(names, char_to_int)
        optimizer.zero_grad()
        loss = 0

        # Initializing the starting state (h or h,c)
        states = model.init_hidden()
        history = [] # For Attention

        for char_idx in range(input_tensor.size(0)):
            x = torch.zeros(1, vocab_size)
            x[0][input_tensor[char_idx]] = 1

            # Logic branch for each architecture
            if isinstance(model, BiLSTM):
                h, c = states
                output, h, c = model(x, h, c)
                states = (h, c)
            elif isinstance(model, AttentionRNN):
                h = states
                # Attention needs the list of all previous 'h' states
                output, h = model(x, h, history)
                states = h
                history.append(h)
            else: # Vanilla RNN
                h = states
                output, h = model(x, h)
                states = h

            loss += criterion(output, target_tensor[char_idx].unsqueeze(0))

        loss.backward()
        # Gradient clipping is helpful for scratch-built RNNs to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if i % 1000 == 0:
            print(f"Epoch {i}, Loss: {loss.item()/input_tensor.size(0):.4f}")

# Generate Name Function
def generate_name(model, char_to_int, int_to_char):
    model.eval()
    with torch.no_grad():
        states = model.init_hidden()
        input_idx = char_to_int['.']
        name = ""
        history = []

        for _ in range(20):
            x = torch.zeros(1, vocab_size)
            x[0][input_idx] = 1

            if isinstance(model, BiLSTM):
                h, c = states
                output, h, c = model(x, h, c)
                states = (h, c)
            elif isinstance(model, AttentionRNN):
                h = states
                output, h = model(x, h, history)
                states = h
                history.append(h)
            else:
                h = states
                output, h = model(x, h)
                states = h

            probs = torch.softmax(output, dim=1)
            input_idx = torch.multinomial(probs, 1).item()

            if input_idx == char_to_int['#']: break
            name += int_to_char[input_idx]

        model.train()
        return name

def evaluate(generated_list, training_list):
    total = len(generated_list)
    unique_gen = set(generated_list)
    training_set = set(training_list)

    novel = [n for n in unique_gen if n not in training_set]
    novelty_rate = (len(novel) / len(unique_gen)) * 100
    diversity = len(unique_gen) / total

    return novelty_rate, diversity

import pandas as pd

def run_experiment(names, vocab, char_to_int, int_to_char):
    all_model_samples = {}
    metrics_records = []

    # Define our three variants from scratch
    models = {
        "Vanilla_RNN": VanillaRNN(len(vocab), 128, len(vocab)),
        "BiLSTM_Scratch": BiLSTM(len(vocab), 128, len(vocab)),
        "RNN_Attention": AttentionRNN(len(vocab), 128, len(vocab))
    }

    print(f"{'Model Architecture':<20} | {'Status':<15}")
    print("-" * 40)

    for name, model in models.items():
        print(f"Training {name}...", end="\r")

        # Train (Standard 3000 epochs for character-level learning)
        train_model(model, names, char_to_int, epochs=3000)

        # Generate 100 names for Task-2 evaluation
        generated = [generate_name(model, char_to_int, int_to_char) for _ in range(100)]
        all_model_samples[name] = generated

        # Task-2: Calculate Metrics
        novelty, diversity = evaluate(generated, names)
        params = sum(p.numel() for p in model.parameters())

        metrics_records.append({
            "Model": name,
            "Trainable_Params": params,
            "Novelty_Rate_Pct": round(novelty, 2),
            "Diversity_Score": round(diversity, 2)
        })
        print(f"{name:<20} | Complete")

    # --- TASK-2 & 3: DISPLAY & SAVE RESULTS ---

    # 1. Prepare the Evaluation Table
    df_metrics = pd.DataFrame(metrics_records)

    print("\n" + "="*50)
    print(" QUANTITATIVE EVALUATION TABLE")
    print("="*50)
    print(df_metrics.to_string(index=False))

    # Save Metrics to TXT
    with open("Evaluation_Metrics.txt", "w") as f:
        f.write("NLP ASSIGNMENT: EVALUATION METRICS\n")
        f.write("="*40 + "\n")
        f.write(df_metrics.to_string(index=False))

    # 2. Display and Save Samples
    print("\n" + "="*50)
    print("generated samples")
    print("="*50)

    with open("Generated_Samples.txt", "w") as f:
        f.write("NLP ASSIGNMENT: GENERATED NAME SAMPLES\n")
        for model_name, sample_list in all_model_samples.items():
            header = f"\n--- MODEL: {model_name} ---"
            print(header)
            f.write(header + "\n")

            # Show first 10 names in console, save all 100 to file
            for i, n in enumerate(sample_list):
                if i < 10: print(f"  {n}")
                f.write(f"{n}\n")

    print("\n" + "="*50)
    print("="*50)

if __name__ == "__main__":
    # Ensure TrainingNames.txt exists in the same folder
    try:
        names, vocab, char_to_int, int_to_char = load_data("TrainingNames.txt")
        run_experiment(names, vocab, char_to_int, int_to_char)
    except FileNotFoundError:
        print("Error: TrainingNames.txt not found.")
