import torch

from training.ablate_history import create_history_ablation
from training.config import TrainConfig
from training.model_history import TichuPolicyHistory, TransformerHistoryEncoder


def _checkpoint(tmp_path, *, split: bool):
    cfg = TrainConfig(
        hidden_dim=32,
        n_layers=1,
        embed_dim=8,
        hist_d_model=16,
        hist_nhead=2,
        hist_n_layers=1,
        hist_ffn_dim=32,
        split_actor_critic_history=split,
    )
    encoder = TransformerHistoryEncoder(
        d_model=cfg.hist_d_model,
        nhead=cfg.hist_nhead,
        n_layers=cfg.hist_n_layers,
        ffn_dim=cfg.hist_ffn_dim,
    )
    policy = TichuPolicyHistory(
        encoder,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        embed_dim=cfg.embed_dim,
        split_actor_critic_history=split,
    )
    with torch.no_grad():
        policy.history_fuse[0].weight.normal_()
        policy.history_fuse[0].bias.normal_()
        if split:
            policy.critic_history_fuse[0].weight.normal_()
            policy.critic_history_fuse[0].bias.normal_()
    source = tmp_path / f"source_{split}.pt"
    output = tmp_path / f"ablated_{split}.pt"
    torch.save(
        {"games_played": 1, "model": policy.state_dict(), "config": cfg},
        source,
    )
    return source, output, cfg.hidden_dim


@torch.no_grad()
def test_history_ablation_preserves_static_fusion_and_bias(tmp_path):
    source, output, hidden_dim = _checkpoint(tmp_path, split=False)
    before = torch.load(source, map_location="cpu", weights_only=False)["model"]

    metadata = create_history_ablation(source, output)
    after = torch.load(output, map_location="cpu", weights_only=False)["model"]

    key = "history_fuse.0.weight"
    assert torch.equal(after[key][:, :hidden_dim], before[key][:, :hidden_dim])
    assert torch.equal(after["history_fuse.0.bias"], before["history_fuse.0.bias"])
    assert torch.count_nonzero(after[key][:, hidden_dim:]).item() == 0
    assert metadata["type"] == "history_input_zero"


@torch.no_grad()
def test_history_ablation_handles_split_actor_critic_fusions(tmp_path):
    source, output, hidden_dim = _checkpoint(tmp_path, split=True)
    before = torch.load(source, map_location="cpu", weights_only=False)["model"]

    metadata = create_history_ablation(source, output)
    after = torch.load(output, map_location="cpu", weights_only=False)["model"]

    for prefix in ("history_fuse.0", "critic_history_fuse.0"):
        weight = f"{prefix}.weight"
        bias = f"{prefix}.bias"
        assert torch.equal(
            after[weight][:, :hidden_dim], before[weight][:, :hidden_dim]
        )
        assert torch.equal(after[bias], before[bias])
        assert torch.count_nonzero(after[weight][:, hidden_dim:]).item() == 0
    assert len(metadata["layers"]) == 2
