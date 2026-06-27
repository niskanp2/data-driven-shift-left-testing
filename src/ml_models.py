import torch
import torch.nn as nn
import random
import numpy as np
import json
import copy

from src.functions import *

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


DIMENSIONS = 4
device = "cuda" if torch.cuda.is_available() else "cpu"

with open(f"gen/embeddings.json", "r") as f:
    embeddings = json.load(f)


# region Helper Functions
def path_to_tensor(path, embeddings=embeddings):
    """Converts a path's embedding to a tensor."""
    sequence_embeddings = [embeddings[node] for node in path]
    return torch.tensor(sequence_embeddings, dtype=torch.float32)


def run_e_step(model, X_templates, y, embeddings, loss_fn, device):
    """
    Evaluates all permutations in X and assigns the most likely path to each template.
    Returns the hardened dataset and the average loss of the best paths.
    """
    model.eval()
    X_hardened = []
    total_loss = 0

    with torch.no_grad():
        for path_template, target in zip(X_templates, y):
            target_tensor = torch.tensor([target], dtype=torch.float32).to(device)
            best_path = None
            best_loss = np.inf

            path_permutations = get_path_permutations(path_template)
            for candidate_path in path_permutations:
                x_tensor = (
                    path_to_tensor(candidate_path, embeddings).unsqueeze(0).to(device)
                )
                length = torch.tensor([len(candidate_path)])

                prediction = model(x_tensor, length)
                loss = loss_fn(prediction.squeeze(), target_tensor.squeeze())

                if loss < best_loss:
                    best_loss = loss
                    best_path = candidate_path

            X_hardened.append(best_path)
            total_loss += best_loss

    avg_loss = total_loss / len(X_templates)
    avg_loss = avg_loss.item()
    return X_hardened, avg_loss


def collate_fn(batch):
    sequences, targets = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded_seqs = pad_sequence(sequences, batch_first=True)
    targets = torch.tensor(targets, dtype=torch.float32)
    return padded_seqs, lengths, targets


# endregion


# region ML model
class GRUModel(nn.Module):
    def __init__(
        self,
        input_size=DIMENSIONS,
        hidden_size=32,
        output_size=1,
        num_layers=1,
        dropout=0.2,
    ):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        """
        x: input of shape (batch_size, seq_length, input_size)
        lengths: actual lengths oof the sequences (batch_size,)
        """
        # Pack the sequences so the GRU ignores padded zeros
        packed_x = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )

        # Forward pass through GRU
        _, h_n = self.gru(packed_x)

        # h_n shape: (num_layers, batch_size, hidden_size)
        # Grab the hidden state from the final layer
        last_output = h_n[-1]

        # Pass through output layer
        output = self.dropout(last_output)
        output = self.fc(output)

        # Squeeze the output to match the target shape (batch_size,)
        return output.squeeze(-1)


class RNNModel(nn.Module):
    def __init__(
        self,
        input_size=DIMENSIONS,
        hidden_size=32,
        output_size=1,
        num_layers=1,
        dropout=0.2,
    ):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Basic RNN layer
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        """
        x: input of shape (batch_size, seq_length, input_size)
        lengths: actual lengths of the sequences (batch_size,)
        """
        # Pack the sequences so the RNN ignores padded zeros
        packed_x = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )

        # Forward pass through basic RNN
        _, h_n = self.rnn(packed_x)

        # h_n shape: (num_layers, batch_size, hidden_size)
        # Grab the hidden state from the final layer
        last_output = h_n[-1]

        # Pass through output layer
        output = self.dropout(last_output)
        output = self.fc(output)

        # Squeeze the output to match the target shape (batch_size,)
        return output.squeeze(-1)

# endregion


# region Datasets
class StochasticDataset(Dataset):
    def __init__(self, X, y, G, embeddings):
        self.X = X
        self.y = y
        self.G = G
        self.embeddings = embeddings

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        path_template = self.X[idx]
        target = self.y[idx]

        current_node = "SPAWN"
        sequence_embeddings = []

        for phase_candidates in path_template:
            # Select node in this phase
            if len(phase_candidates) == 1:
                chosen_node = phase_candidates[0]
            else:
                # Fetch edge weights from current_node to each candidate
                weights = []
                for candidate_node in phase_candidates:
                    weight = self.G[current_node][candidate_node]["weight"]
                    weights.append(weight)
                chosen_node = random.choices(phase_candidates, weights=weights, k=1)[0]

            # Fetch embedding for chosen node
            emb = self.embeddings[chosen_node]
            sequence_embeddings.append(emb)

            # Update current_node for next transition
            current_node = chosen_node

        # Format data as tensors
        x_tensor = torch.tensor(sequence_embeddings, dtype=torch.float32)
        y_tensor = torch.tensor(target, dtype=torch.float32)

        return x_tensor, y_tensor


class EMDataset(Dataset):
    def __init__(self, X, y, embeddings):
        self.X = X
        self.y = y
        self.embeddings = embeddings

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        path = self.X[idx]
        target = self.y[idx]

        x_tensor = path_to_tensor(path, self.embeddings)
        y_tensor = torch.tensor(target, dtype=torch.float32)

        return x_tensor, y_tensor


# endregion


# region Training Loops
def train_em_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    G,
    embeddings,
    init_epochs=5,
    m_step_epochs=20,
    max_epochs=200,
    batch_size=32,
    patience=20,
    m_patience=5,
    lr=0.001,
    device="cpu",
):
    assert max_epochs > init_epochs, f"Max epochs must be greater than init epochs."
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Stochastic initialization
    if init_epochs > 0:
        print(f"Starting Stochastic Initialization ({init_epochs} epochs)")
        init_dataset = StochasticDataset(X_train, y_train, G, embeddings)
        init_loader = DataLoader(
            init_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )

        for _ in range(init_epochs):
            model.train()
            for batch_X, batch_lengths, batch_y in init_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                predictions = model(batch_X, batch_lengths)
                loss = loss_fn(predictions.squeeze(), batch_y.squeeze())
                loss.backward()
                optimizer.step()

    # EM Loop
    print("Starting EM Loop")
    current_epoch = 0
    em_iteration = 1

    # Track the best model based on validation performance
    best_model_state = copy.deepcopy(model.state_dict())
    best_val_loss = np.inf
    best_X_train = None
    best_X_test = None

    iterations_since_improvement = 0
    e_losses = []
    m_losses = []
    val_losses = []

    while current_epoch < max_epochs:
        # --- TRAINING ---
        # E-step
        X_hardened_train, e_loss = run_e_step(
            model, X_train, y_train, embeddings, loss_fn, device
        )
        e_losses.append(e_loss)

        # M-step
        m_dataset = EMDataset(X_hardened_train, y_train, embeddings)
        m_loader = DataLoader(
            m_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )

        last_m_epoch_loss = 0
        best_m_epoch_loss = np.inf
        epochs_since_improvement = 0
        for _ in range(m_step_epochs):
            model.train()
            epoch_loss = 0
            for batch_X, batch_lengths, batch_y in m_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                predictions = model(batch_X, batch_lengths)
                loss = loss_fn(predictions.squeeze(), batch_y.squeeze())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            mean_m_epoch_loss = epoch_loss / len(m_loader)
            last_m_epoch_loss = mean_m_epoch_loss
            current_epoch += 1

            # Run M-Step Early Stopping
            if mean_m_epoch_loss < best_m_epoch_loss:
                best_m_epoch_loss = mean_m_epoch_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if epochs_since_improvement >= m_patience:
                break

        m_losses.append(last_m_epoch_loss)

        # --- VALIDATION ---
        X_hardened_test, current_val_loss = run_e_step(
            model, X_test, y_test, embeddings, loss_fn, device
        )
        val_losses.append(current_val_loss)

        # --- CHECKPOINTING ---
        if em_iteration % 10 == 0:
            print(f"[Iteration {em_iteration: >3}] - Val Loss: {current_val_loss:.4f}")

        # Run Early Stopping
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_X_train = X_hardened_train
            best_X_test = X_hardened_test
            iterations_since_improvement = 0
        else:
            iterations_since_improvement += 1

        if iterations_since_improvement >= patience:
            print("Early stopping triggered.")
            break

        em_iteration += 1

    # Load best model weights before returning
    model.load_state_dict(best_model_state)

    print(f"Best Validation Loss: {best_val_loss:.4f}")

    return_dict = {
        "model": model,
        "X_train": best_X_train,
        "X_test": best_X_test,
        "e_losses": e_losses,
        "m_losses": m_losses,
        "val_losses": val_losses,
        "best_loss": best_val_loss,
    }

    return return_dict


# endregion
