"""Smoke tests: run full games with random legal actions to verify stability."""

import random

import numpy as np
import torch

from tichu_env.cards import MAHJONG_INDEX
from tichu_env.encoding import (
    PLAY_TOKEN_SIZE,
    STATIC_OBS_SIZE,
    encode_history,
    encode_pass_source_ids,
    encode_pass_target_ids,
    encode_static,
)
from tichu_env.game_state import GameState
from tichu_env.env import TichuEnv
from training.model_embed import ACTION_DIM
from training.model_history import TichuPolicyHistory, TransformerHistoryEncoder
from training.rollout_embed import _action_vec
from training.rollout_history import ROLLOUT_FORWARD_BUCKET


class TestGameStateRollout:
    """Test raw GameState with random plays."""

    def _play_random_game(self, seed: int) -> GameState:
        rng = random.Random(seed)
        gs = GameState()
        gs.reset(seed=seed)

        max_steps = 500  # Safety limit
        steps = 0

        while not gs.round_over and steps < max_steps:
            steps += 1

            # Handle dragon give
            if gs.dragon_give_pending is not None:
                player = gs.dragon_give_pending
                options = gs.get_dragon_give_options(player)
                assert len(options) > 0, f"No dragon give options for player {player}"
                recipient = rng.choice(options)
                gs.give_dragon_trick(player, recipient)
                continue

            player = gs.current_player
            actions = gs.get_legal_actions(player)
            assert len(actions) > 0, (
                f"No legal actions for player {player}, "
                f"hand={gs.hands[player]}, leading={gs.trick.leading_combo}"
            )

            action = rng.choice(actions)
            success = gs.play(player, action)
            assert success, (
                f"Legal action failed: player={player}, action={action}, "
                f"leading={gs.trick.leading_combo}"
            )

            # Set mahjong wish randomly if mahjong was just played
            if gs.mahjong_wish is None and any(
                52 in cards for _, cards, _ in gs.trick_history[-1:]
            ):
                # Random wish or no wish
                if rng.random() < 0.5:
                    gs.set_mahjong_wish(rng.randint(2, 14))

        assert gs.round_over, f"Game didn't finish in {max_steps} steps (seed={seed})"
        return gs

    def test_single_game(self):
        gs = self._play_random_game(42)
        a, b = gs.round_scores
        # Scores should be assigned
        assert isinstance(a, int) and isinstance(b, int)

    def test_many_games(self):
        """Run 50 random games — none should crash."""
        for seed in range(50):
            gs = self._play_random_game(seed)
            assert gs.round_over

    def test_all_cards_accounted_for(self):
        """After a game, all 56 cards should be in won_cards or last player's hand."""
        gs = self._play_random_game(123)
        all_won = set()
        for cards in gs.won_cards:
            all_won |= set(cards)
        # All cards should be won (since round is over, all cards distributed)
        assert len(all_won) == 56, f"Only {len(all_won)} cards accounted for"

    def test_finish_order_complete(self):
        gs = self._play_random_game(99)
        assert len(gs.finish_order) == 4
        assert set(gs.finish_order) == {0, 1, 2, 3}


class TestEnvRollout:
    """Test PettingZoo env with random actions."""

    def _play_random_env_game(self, seed: int):
        env = TichuEnv(seed=seed)
        env.reset(seed=seed)

        max_steps = 600
        steps = 0

        while not all(env.terminations.values()) and steps < max_steps:
            steps += 1
            agent = env.agent_selection
            obs, reward, term, trunc, info = env.last()

            if term or trunc:
                # Dead agent, step with None action
                env.step(0)
                continue

            mask = obs["action_mask"]
            legal = [i for i in range(len(mask)) if mask[i]]
            assert len(legal) > 0, f"No legal actions for {agent}"

            action = random.choice(legal)
            env.step(action)

        assert all(env.terminations.values()), f"Env didn't terminate in {max_steps} steps"
        return env

    def test_single_env_game(self):
        env = self._play_random_env_game(42)
        # Check rewards are assigned
        for agent in env.possible_agents:
            assert isinstance(env._cumulative_rewards[agent], float)

    def test_many_env_games(self):
        """Run 20 random env games."""
        for seed in range(20):
            self._play_random_env_game(seed)

    def test_observation_shape_static_only(self):
        env = TichuEnv(seed=0)
        env.reset(seed=0)
        obs, _, _, _, _ = env.last()
        assert obs["observation"].shape == (STATIC_OBS_SIZE,)
        assert obs["pass_source_ids"].shape == (3,)
        assert obs["pass_target_ids"].shape == (3,)
        assert obs["action_mask"].shape == (2000,)

    def test_observation_shape_with_history(self):
        env = TichuEnv(seed=0, track_history=True)
        env.reset(seed=0)
        obs, _, _, _, _ = env.last()
        assert obs["static"].shape == (STATIC_OBS_SIZE,)
        assert obs["pass_source_ids"].shape == (3,)
        assert obs["pass_target_ids"].shape == (3,)
        assert obs["history"].shape == (200, 63)
        assert isinstance(obs["history_len"], int)
        assert obs["action_mask"].shape == (2000,)

    def test_observation_values_bounded(self):
        env = TichuEnv(seed=0)
        env.reset(seed=0)
        obs, _, _, _, _ = env.last()
        assert obs["observation"].min() >= 0.0
        assert obs["observation"].max() <= 1.5  # cards_remaining can be 14/14=1.0

    def test_rewards_symmetric(self):
        """Team A and Team B rewards should be opposite."""
        env = self._play_random_env_game(42)
        r0 = env._cumulative_rewards["player_0"]
        r1 = env._cumulative_rewards["player_1"]
        r2 = env._cumulative_rewards["player_2"]
        r3 = env._cumulative_rewards["player_3"]
        # Team A (0, 2) should have same reward
        assert r0 == r2
        # Team B (1, 3) should have same reward
        assert r1 == r3
        # Opposite
        assert abs(r0 + r1) < 1e-6


class TestHistoryRolloutBucketing:
    def test_bucketed_forward_matches_unbucketed(self):
        rng = random.Random(123)
        torch.manual_seed(123)
        np.random.seed(123)

        policy = TichuPolicyHistory(TransformerHistoryEncoder())
        policy.eval()

        games = [GameState(track_history=True) for _ in range(24)]
        for i, gs in enumerate(games):
            gs.reset(seed=1000 + i)

        for _ in range(15):
            for gs in games:
                if gs.round_over:
                    continue
                while gs.dragon_give_pending is not None and not gs.round_over:
                    opts = gs.get_dragon_give_options(gs.dragon_give_pending)
                    gs.give_dragon_trick(gs.dragon_give_pending, rng.choice(opts))
                if gs.round_over:
                    continue
                player = gs.current_player
                legal = gs.get_policy_legal_actions(player)
                action = rng.choice(legal)
                gs.play(player, action)
                if MAHJONG_INDEX in action[0] and rng.random() < 0.5:
                    gs.set_mahjong_wish(rng.randint(2, 14))

        active = []
        for i, gs in enumerate(games):
            while gs.dragon_give_pending is not None and not gs.round_over:
                options = gs.get_dragon_give_options(gs.dragon_give_pending)
                gs.give_dragon_trick(
                    gs.dragon_give_pending, rng.choice(options)
                )
            if gs.round_over:
                continue
            player = gs.current_player
            legal = gs.get_policy_legal_actions(player)
            obs = encode_static(gs, player)
            pass_src = encode_pass_source_ids(gs, player)
            pass_tgt = encode_pass_target_ids(gs, player)
            vecs = np.array([_action_vec(a) for a in legal], dtype=np.float32)
            hist, hist_len = encode_history(gs, player)
            active.append((i, player, obs, pass_src, pass_tgt, vecs, hist, hist_len))

        n_legal = np.array([x[5].shape[0] for x in active], dtype=np.int32)
        max_n = int(n_legal.max())
        legal_pad = np.zeros((len(active), max_n, ACTION_DIM), dtype=np.float32)
        for j, (_, _, _, _, _, vecs, _, _) in enumerate(active):
            legal_pad[j, : len(vecs)] = vecs

        hist_lens = np.array([x[7] for x in active], dtype=np.int32)
        max_hist = int(hist_lens.max()) if hist_lens.max() > 0 else 1
        hist_batch = np.zeros((len(active), max_hist, PLAY_TOKEN_SIZE), dtype=np.float32)
        for j, (_, _, _, _, _, _, hist, _) in enumerate(active):
            hist_batch[j] = hist[:max_hist]

        with torch.no_grad():
            ref_logits, ref_values = policy._forward_scores(
                torch.as_tensor(np.stack([x[2] for x in active]), dtype=torch.float32),
                torch.as_tensor(legal_pad),
                torch.as_tensor(hist_batch),
                torch.as_tensor(hist_lens),
                torch.as_tensor(np.stack([x[3] for x in active]), dtype=torch.int64),
                torch.as_tensor(np.stack([x[4] for x in active]), dtype=torch.int64),
            )

        order = sorted(
            range(len(active)),
            key=lambda idx: (active[idx][7], active[idx][5].shape[0]),
        )
        got_logits = [None] * len(active)
        got_values = np.zeros(len(active), dtype=np.float32)

        for start in range(0, len(order), ROLLOUT_FORWARD_BUCKET):
            bucket_idx = order[start : start + ROLLOUT_FORWARD_BUCKET]
            bucket = [active[idx] for idx in bucket_idx]
            bucket_n_legal = np.array([x[5].shape[0] for x in bucket], dtype=np.int32)
            bucket_max_n = int(bucket_n_legal.max())
            bucket_legal_pad = np.zeros(
                (len(bucket), bucket_max_n, ACTION_DIM), dtype=np.float32
            )
            for j, (_, _, _, _, _, vecs, _, _) in enumerate(bucket):
                bucket_legal_pad[j, : len(vecs)] = vecs

            bucket_hist_lens = np.array([x[7] for x in bucket], dtype=np.int32)
            bucket_max_hist = int(bucket_hist_lens.max()) if bucket_hist_lens.max() > 0 else 1
            bucket_hist = np.zeros(
                (len(bucket), bucket_max_hist, PLAY_TOKEN_SIZE), dtype=np.float32
            )
            for j, (_, _, _, _, _, _, hist, _) in enumerate(bucket):
                bucket_hist[j] = hist[:bucket_max_hist]

            with torch.no_grad():
                logits, values = policy._forward_scores(
                    torch.as_tensor(np.stack([x[2] for x in bucket]), dtype=torch.float32),
                    torch.as_tensor(bucket_legal_pad),
                    torch.as_tensor(bucket_hist),
                    torch.as_tensor(bucket_hist_lens),
                    torch.as_tensor(np.stack([x[3] for x in bucket]), dtype=torch.int64),
                    torch.as_tensor(np.stack([x[4] for x in bucket]), dtype=torch.int64),
                )

            for j, idx in enumerate(bucket_idx):
                got_logits[idx] = logits[j, : bucket_n_legal[j]].cpu().numpy()
                got_values[idx] = float(values[j].item())

        for i, n in enumerate(n_legal):
            ref = ref_logits[i, :n].cpu().numpy()
            np.testing.assert_allclose(ref, got_logits[i], atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(
            ref_values.cpu().numpy(), got_values, atol=1e-7, rtol=1e-7
        )
