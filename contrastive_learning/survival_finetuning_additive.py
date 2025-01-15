import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, KFold
import torchtuples as tt
from pycox.models import CoxPH
from sksurv.metrics import concordance_index_censored  # Using sksurv instead of lifelines
import warnings

# Suppress specific pandas warnings
warnings.filterwarnings("ignore", message=".*numexpr.*")

# ---------------------------------------------------------------------------
# 1) Define the Additive Attention Aggregator Model
# ---------------------------------------------------------------------------
class AdditiveAttentionAggregator(nn.Module):
    """
    Turns [batch_size, N, input_dim] into [batch_size, input_dim]
    by learning attention weights for each of the N embeddings.
    """
    def __init__(self, input_dim=192, attention_dim=256):
        super(AdditiveAttentionAggregator, self).__init__()
        # This MLP produces a single attention logit per tile
        self.attention_mlp = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )

    def forward(self, x):
        """
        Args:
          x: [batch_size, N, input_dim]
        Returns:
          out: [batch_size, input_dim] — weighted sum of embeddings
        """
        if x.isnan().any():
            raise ValueError("NaNs detected in input embeddings.")

        # Compute attention logits: [batch_size, N, 1]
        attn_logits = self.attention_mlp(x)

        # Convert logits to normalized weights (softmax across N)
        attn_weights = torch.softmax(attn_logits, dim=1)  # [batch_size, N, 1]

        # Weighted sum of embeddings using the learned attention
        out = (x * attn_weights).sum(dim=1)  # [batch_size, input_dim]

        return out

# ---------------------------------------------------------------------------
# 2) Define the Combined HIPT Aggregator + Cox Model
# ---------------------------------------------------------------------------
class HiptAggregatorCox(nn.Module):
    """
    Combined Model:
    - AdditiveAttentionAggregator: Nx192 -> 1x192
    - Small MLP for log-hazard prediction
    """
    def __init__(self, aggregator: nn.Module, hidden_dim=256):
        super(HiptAggregatorCox, self).__init__()
        self.aggregator = aggregator
        self.cox_mlp = nn.Sequential(
            nn.Linear(192, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Single log-hazard output
        )

    def forward(self, x):
        """
        x: [batch_size, N, 192]
        Returns [batch_size, 1] log-hazard
        """
        x = self.aggregator(x)  # [batch_size, 192]
        if x.isnan().any():
            raise ValueError("NaNs detected after aggregation.")
        hazard = self.cox_mlp(x)  # [batch_size, 1]
        return hazard

# ---------------------------------------------------------------------------
# 3) Define the Dataset
# ---------------------------------------------------------------------------
class SurvivalTilesDataset(Dataset):
    """
    Each item:
      - Nx192 embeddings
      - event indicator (0/1)
      - duration (float)
    """
    def __init__(self, embeddings, events, times):
        self.embeddings = [np.ascontiguousarray(e, dtype=np.float32) for e in embeddings]
        self.events = np.array(events, dtype=np.float32)
        self.times = np.array(times, dtype=np.float32)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.embeddings[idx])   # [N, 192]
        evt = torch.tensor(self.events[idx], dtype=torch.float32)
        dur = torch.tensor(self.times[idx], dtype=torch.float32)
        return x, (evt, dur)

# ---------------------------------------------------------------------------
# 4) Helper Function to Predict Hazards
# ---------------------------------------------------------------------------
def predict_hazard(cox_model, data_loader, device):
    """
    Predict hazard scores using the Pycox CoxPH model (which wraps our aggregator).
    """
    cox_model.net.eval()
    preds = []
    with torch.no_grad():
        for (x_batch, _) in data_loader:
            x_batch = x_batch.to(device)
            hazard = cox_model.net(x_batch)  # [batch_size, 1]
            if torch.isnan(hazard).any():
                raise ValueError("NaNs detected in hazard predictions.")
            preds.append(hazard.cpu().numpy())
    # Combine all predictions into a single array of shape [num_samples]
    preds = np.vstack(preds).squeeze()
    return preds

# ---------------------------------------------------------------------------
# 5) Device Setup
# ---------------------------------------------------------------------------
def setup_device():
    """
    Pick 'cuda' if available, else CPU. If multiple GPUs, use 'cuda:0'.
    """
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            device = torch.device('cuda:0')
            print(f"Using {n_gpus} GPUs")
        else:
            device = torch.device('cuda')
            print("Using a single GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

# ---------------------------------------------------------------------------
# 6) Main Function with Nested Cross-Validation
# ---------------------------------------------------------------------------
def main():
    # Update paths if necessary
    embeddings_folder = "scratch/TCGA-BLCA-embeddings"
    metadata_file = "svs_patient_map_PFI_blca.json"

    # 1) Load metadata
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    # 2) Gather Nx192 embeddings + events + durations
    embeddings, events, durations = [], [], []
    for entry in metadata:
        file_name = entry["file_name"].replace(".svs", ".pt")
        embedding_path = os.path.join(embeddings_folder, file_name)

        if os.path.exists(embedding_path):
            try:
                tile_tensor = torch.load(embedding_path, map_location='cpu')  # shape [N,192] ideally
                if tile_tensor.dim() != 2 or tile_tensor.size(1) != 192:
                    print(f"Skipping {file_name}: Expected shape [N, 192], got {tile_tensor.shape}")
                    continue
                embeddings.append(tile_tensor.numpy())
                events.append(entry["event"])       # 0 or 1
                durations.append(entry["time"])     # float
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
                continue
        else:
            print(f"File not found: {embedding_path}")
            continue

    # Convert to proper numpy arrays
    events = np.array(events, dtype=np.float32)
    durations = np.array(durations, dtype=np.float32)

    # Basic sanity check
    if np.isnan(events).any() or np.isnan(durations).any():
        raise ValueError("NaNs detected in events or durations. Please clean your data.")

    # 3) Device
    device = setup_device()

    # 4) Nested Cross-Validation Setup
    outer_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    outer_cindices = []
    print("\nStarting Nested CV for HIPT Aggregator + Cox Model...\n")

    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(
        outer_kf.split(embeddings, events), start=1
    ):
        print(f"Outer Fold {outer_fold}/5")

        # Split data
        X_trainval = [embeddings[i] for i in outer_train_idx]
        X_test     = [embeddings[i] for i in outer_test_idx]
        E_trainval = events[outer_train_idx]
        E_test     = events[outer_test_idx]
        T_trainval = durations[outer_train_idx]
        T_test     = durations[outer_test_idx]

        best_inner_cindex = -np.inf
        best_model = None

        # Inner CV for hyperparam tuning
        for inner_fold, (inner_tr_idx, inner_val_idx) in enumerate(inner_kf.split(X_trainval, E_trainval), start=1):
            print(f"  Inner Fold {inner_fold}/3 of Outer Fold {outer_fold}")

            # Prepare data
            X_inner_train = [X_trainval[i] for i in inner_tr_idx]
            X_inner_val   = [X_trainval[i] for i in inner_val_idx]
            E_inner_train = E_trainval[inner_tr_idx]
            E_inner_val   = E_trainval[inner_val_idx]
            T_inner_train = T_trainval[inner_tr_idx]
            T_inner_val   = T_trainval[inner_val_idx]

            # Datasets & Loaders
            ds_inner_train = SurvivalTilesDataset(X_inner_train, E_inner_train, T_inner_train)
            ds_inner_val   = SurvivalTilesDataset(X_inner_val,   E_inner_val,   T_inner_val)

            dl_inner_train = DataLoader(ds_inner_train, batch_size=1, shuffle=True)
            dl_inner_val   = DataLoader(ds_inner_val,   batch_size=1, shuffle=False)

            # Model (use our AdditiveAttentionAggregator)
            aggregator = AdditiveAttentionAggregator(input_dim=192, attention_dim=256).to(device)
            combined_model = HiptAggregatorCox(aggregator, hidden_dim=256).to(device)

            # Optimizer & CoxPH
            optimizer = tt.optim.Adam(lr=1e-3)
            cox_ph = CoxPH(combined_model, optimizer)

            # Train
            try:
                cox_ph.fit_dataloader(dl_inner_train, epochs=50, verbose=False)
            except Exception as e:
                print(f"  Error during training: {e}")
                continue

            # Predict
            try:
                val_preds = predict_hazard(cox_ph, dl_inner_val, device)
            except ValueError as ve:
                print(f"  Prediction error: {ve}")
                continue

            # Compute C-index via sksurv
            try:
                cindex_inner = concordance_index_censored(
                    E_inner_val.astype(bool),  # True/False
                    T_inner_val,
                    -val_preds                # negative hazard => higher = more risk
                )[0]  # Extract only the cindex
            except Exception as e:
                print(f"  Error computing C-index: {e}")
                continue

            print(f"    Inner Fold {inner_fold} C-Index: {cindex_inner:.4f}")

            # Track best
            if cindex_inner > best_inner_cindex:
                best_inner_cindex = cindex_inner
                best_model = cox_ph

        # Evaluate on outer test
        if best_model is not None:
            ds_test = SurvivalTilesDataset(X_test, E_test, T_test)
            dl_test = DataLoader(ds_test, batch_size=1, shuffle=False)
            try:
                test_preds = predict_hazard(best_model, dl_test, device)
                if np.isnan(test_preds).any():
                    print(f"  Warning: NaNs in test predictions for Outer Fold {outer_fold}. Skipping.")
                    continue
                cindex_outer = concordance_index_censored(
                    E_test.astype(bool),
                    T_test,
                    -test_preds
                )[0]  # Extract only the cindex
                print(f"  Outer Fold {outer_fold} Test C-Index: {cindex_outer:.4f}")
                outer_cindices.append(cindex_outer)
            except Exception as e:
                print(f"  Error during outer test evaluation: {e}")
                continue
        else:
            print(f"  No valid inner model found for Outer Fold {outer_fold}. Skipping test evaluation.")

    # Final results
    if outer_cindices:
        mean_cidx = np.mean(outer_cindices)
        std_cidx  = np.std(outer_cindices)
        print("\n=== Final Nested CV Results ===")
        print("Outer-Fold C-Indices:", outer_cindices)
        print(f"Mean C-Index = {mean_cidx:.4f}, Std = {std_cidx:.4f}")
    else:
        print("No valid outer fold evaluations were completed.")

# ---------------------------------------------------------------------------
# 7) Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
