"""Train a 'Tiny History' model locally, starting from the 1M embed_tiny model."""

from __future__ import annotations
import multiprocessing as mp
import torch
import os

from training.config import TrainConfig
from training.train_history import train, init_from_embed_checkpoint, _build_policy

def main():
    # 1. Define Tiny History Config (matching embed_tiny dimensions)
    cfg = TrainConfig(
        checkpoint_dir="checkpoints/tiny_history_fresh",
        log_file="logs/train_tiny_history_fresh.csv",
        hidden_dim=128,    # Matches embed_tiny
        n_layers=2,        # Matches embed_tiny
        embed_dim=64,
        hist_d_model=64,   
        hist_nhead=2,      
        hist_n_layers=2,   
        hist_ffn_dim=128,  
        lr=3e-4,
        batch_size=1024,
        games_per_update=256,
        n_workers=min(4, os.cpu_count()),
        device="cpu",      # Training locally
        synchronous=True,  # Force Synchronous PPO
    )

    latest_path = "checkpoints/tiny_history_fresh/latest.pt"
    
    # Check if we should resume
    if os.path.exists(latest_path):
        print(f"Resuming Tiny History from {latest_path}...")
        train(cfg, resume=latest_path)
    else:
        print("Starting Tiny History from SCRATCH (Game 0)...")
        train(cfg)

if __name__ == "__main__":
    # Required for multiprocessing on some systems
    mp.set_start_method("spawn", force=True)
    main()
