"""
Anomaly model training script.
Run after collecting baseline data:
  python3 src/anomaly/train.py

Reads scene states from Redis replay buffer,
trains Isolation Forest + LSTM Autoencoder,
saves models to models/anomaly/
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
import redis

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anomaly.features import (
    scene_states_to_feature_matrix,
    FEATURE_DIM, FEATURE_NAMES
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train")

# ── Config ────────────────────────────────────────────────────────────────────

REDIS_HOST      = os.getenv("REDIS_HOST",     "localhost")
REDIS_PASSWORD  = os.getenv("REDIS_PASSWORD", "")
REDIS_PORT      = int(os.getenv("REDIS_PORT", 6379))
MODEL_DIR       = "models/anomaly"
MIN_SAMPLES     = 100    # minimum feature vectors to train

# ── Load scene states from Redis ──────────────────────────────────────────────

def load_scene_states_from_redis(r: redis.Redis) -> list[dict]:
    logger.info("Loading scene states from Redis replay buffer...")
    raw_states = r.lrange("scene:replay_buffer", 0, -1)
    logger.info(f"Found {len(raw_states)} frames in replay buffer")

    states = []
    for raw in raw_states:
        try:
            states.append(json.loads(raw))
        except json.JSONDecodeError:
            continue

    # Filter to only live camera frames (not stubs)
    live = [s for s in states if s.get("source") == "live_yolov8"]
    logger.info(f"Live YOLOv8 frames: {len(live)}")
    return live

# ── Isolation Forest ──────────────────────────────────────────────────────────

def train_isolation_forest(X: np.ndarray) -> object:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    logger.info(f"Training Isolation Forest on {len(X)} samples...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.02,
        max_samples='auto',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)

    # Compute baseline score statistics on training data
    scores = -model.decision_function(X_scaled)
    threshold = float(np.percentile(scores, 98))  # 98th percentile = anomaly

    logger.info(f"Isolation Forest trained")
    logger.info(f"Score range: {scores.min():.3f} – {scores.max():.3f}")
    logger.info(f"Threshold (98th pct): {threshold:.3f}")

    return {
        "model":     model,
        "scaler":    scaler,
        "threshold": threshold,
        "score_mean": float(scores.mean()),
        "score_std":  float(scores.std())
    }

# ── LSTM Autoencoder ──────────────────────────────────────────────────────────

def build_sequences(
    X: np.ndarray,
    window_size: int = 20,
    step: int = 5
) -> np.ndarray:
    """Build sliding window sequences from feature matrix"""
    sequences = []
    for i in range(0, len(X) - window_size, step):
        sequences.append(X[i:i + window_size])
    return np.array(sequences)

def train_lstm_autoencoder(
    X: np.ndarray,
    window_size: int = 20,
    epochs: int = 50
) -> dict:
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler

    logger.info(f"Training LSTM Autoencoder on {len(X)} samples...")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sequences = build_sequences(X_scaled, window_size)
    if len(sequences) < 10:
        logger.warning("Not enough sequences for LSTM training — "
                       "need more data. Skipping LSTM.")
        return None

    logger.info(f"Built {len(sequences)} sequences of length {window_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")

    # ── Model definition ──────────────────────────────────────────────────────
    class LSTMAutoencoder(nn.Module):
        def __init__(self, input_dim, hidden_dim=32, latent_dim=8):
            super().__init__()
            self.encoder = nn.LSTM(
                input_dim, hidden_dim,
                num_layers=1, batch_first=True
            )
            self.enc_fc  = nn.Linear(hidden_dim, latent_dim)
            self.dec_fc  = nn.Linear(latent_dim, hidden_dim)
            self.decoder = nn.LSTM(
                hidden_dim, input_dim,
                num_layers=1, batch_first=True
            )

        def forward(self, x):
            _, (h, _) = self.encoder(x)
            latent    = self.enc_fc(h[-1])
            dec_init  = self.dec_fc(latent)
            dec_init  = dec_init.unsqueeze(1).repeat(1, x.size(1), 1)
            out, _    = self.decoder(dec_init)
            return out

        def reconstruction_error(self, x):
            with torch.no_grad():
                recon = self.forward(x)
                return torch.mean((recon - x) ** 2, dim=(1, 2))

    model     = LSTMAutoencoder(FEATURE_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn   = nn.MSELoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    dataset = torch.FloatTensor(sequences).to(device)

    for epoch in range(epochs):
        model.train()
        # Mini-batch
        idx      = torch.randperm(len(dataset))
        batch    = dataset[idx[:min(64, len(dataset))]]
        recon    = model(batch)
        loss     = loss_fn(recon, batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % 10 == 0:
            logger.info(f"  Epoch {epoch:3d}/{epochs} loss={loss.item():.6f}")

    # ── Baseline reconstruction error ─────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        errors    = model.reconstruction_error(dataset).cpu().numpy()
    threshold = float(np.percentile(errors, 98))

    logger.info(f"LSTM Autoencoder trained")
    logger.info(f"Reconstruction error range: "
                f"{errors.min():.4f} – {errors.max():.4f}")
    logger.info(f"Threshold (98th pct): {threshold:.4f}")

    return {
        "model":     model,
        "scaler":    scaler,
        "threshold": threshold,
        "window_size": window_size,
        "error_mean":  float(errors.mean()),
        "error_std":   float(errors.std())
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Connect to Redis
    r = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        decode_responses=True
    )
    r.ping()
    logger.info("Redis connected")

    # Load and convert scene states
    states = load_scene_states_from_redis(r)
    if not states:
        logger.error("No live scene states found in replay buffer")
        sys.exit(1)

    X, entity_ids = scene_states_to_feature_matrix(states)
    logger.info(f"Feature matrix: {X.shape} "
                f"({X.shape[0]} observations, {X.shape[1]} features)")

    if len(X) < MIN_SAMPLES:
        logger.warning(
            f"Only {len(X)} samples — need at least {MIN_SAMPLES}. "
            f"Let the pipeline run longer and retry."
        )
        # Continue anyway for demo purposes
        if len(X) == 0:
            sys.exit(1)

    # Log feature statistics
    logger.info("\nFeature statistics:")
    import pandas as pd
    from anomaly.features import FEATURE_NAMES
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    logger.info(f"\n{df.describe().to_string()}")

    # Train Isolation Forest
    iforest_bundle = train_isolation_forest(X)
    with open(f"{MODEL_DIR}/iforest.pkl", "wb") as f:
        pickle.dump(iforest_bundle, f)
    logger.info(f"Saved Isolation Forest to {MODEL_DIR}/iforest.pkl")

    # Train LSTM Autoencoder
    lstm_bundle = train_lstm_autoencoder(X)
    if lstm_bundle:
        import torch
        # Save model weights separately from non-serializable objects
        torch.save(
            lstm_bundle["model"].state_dict(),
            f"{MODEL_DIR}/lstm_weights.pt"
        )
        # Save everything except the model object
        lstm_meta = {k: v for k, v in lstm_bundle.items() if k != "model"}
        lstm_meta["scaler"] = lstm_bundle["scaler"]
        with open(f"{MODEL_DIR}/lstm_meta.pkl", "wb") as f:
            pickle.dump(lstm_meta, f)
        logger.info(f"Saved LSTM to {MODEL_DIR}/lstm_weights.pt")

    # Save training metadata
    meta = {
        "trained_at":      __import__('datetime').datetime.now().isoformat(),
        "num_samples":     len(X),
        "num_frames":      len(states),
        "feature_names":   FEATURE_NAMES,
        "iforest_threshold": iforest_bundle["threshold"],
        "lstm_threshold":    lstm_bundle["threshold"] if lstm_bundle else None,
    }
    with open(f"{MODEL_DIR}/training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"\n{'='*50}")
    logger.info(f"Training complete")
    logger.info(f"Samples:  {len(X)}")
    logger.info(f"IForest threshold:  {iforest_bundle['threshold']:.3f}")
    if lstm_bundle:
        logger.info(f"LSTM threshold:     {lstm_bundle['threshold']:.4f}")
    logger.info(f"Models saved to {MODEL_DIR}/")
    logger.info(f"{'='*50}")

if __name__ == "__main__":
    main()