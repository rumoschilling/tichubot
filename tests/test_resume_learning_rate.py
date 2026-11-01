import torch

from training.train_history import _load_optimizer_keep_lr


def _adam(lr):
    params = [torch.nn.Parameter(torch.zeros(3)), torch.nn.Parameter(torch.zeros(2))]
    return torch.optim.Adam([{"params": params[:1], "lr": lr}, {"params": params[1:], "lr": lr * 2}])


def test_resume_keeps_configured_learning_rate_and_moments():
    saved = _adam(3e-4)
    for group in saved.param_groups:
        for p in group["params"]:
            p.grad = torch.ones_like(p)
    saved.step()
    state = saved.state_dict()

    resumed = _adam(6e-4)
    _load_optimizer_keep_lr(resumed, state, "policy")

    assert [g["lr"] for g in resumed.param_groups] == [6e-4, 12e-4]
    first = resumed.param_groups[0]["params"][0]
    assert torch.equal(resumed.state[first]["exp_avg"], saved.state[saved.param_groups[0]["params"][0]]["exp_avg"])
