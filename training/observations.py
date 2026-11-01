"""Select the static observation variant expected by a loaded policy."""

from tichu_env.encoding import (
    COUNTED_TRICK_POINTS_OBS_SIZE,
    STATIC_OBS_SIZE,
    encode_static,
    encode_static_counted_trick_points,
)


def encode_policy_static(policy, game_state, observer: int):
    obs_size = getattr(policy, "obs_size", STATIC_OBS_SIZE)
    if obs_size == STATIC_OBS_SIZE:
        return encode_static(game_state, observer)
    if obs_size == COUNTED_TRICK_POINTS_OBS_SIZE:
        return encode_static_counted_trick_points(game_state, observer)
    raise ValueError(f"Unsupported policy observation size: {obs_size}")
