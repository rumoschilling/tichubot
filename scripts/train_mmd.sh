#!/usr/bin/env bash
# Train a TichuBot model from scratch with the recipe behind the released
# model: a compact history model, PPO self-play, and Magnetic Mirror Descent
# (moving magnet mixed with uniform) on the play policy.
#
#   scripts/train_mmd.sh                 # writes to runs/mmd
#   OUT=runs/try2 GAMES=5000000 scripts/train_mmd.sh
#
# Needs a CUDA GPU with about 9 GB free (set DEVICE=cpu to try it slowly).
# The released model had 36.6M games, about a week on one RTX 3080.
# The run resumes from its last checkpoint after a crash (at most 50
# attempts). Stop it with Ctrl-C, or `kill -INT <python pid>`; a checkpoint
# is written every CKPT_EVERY games.
set -u
OUT=${OUT:-runs/mmd}
GAMES=${GAMES:-500000000}
DEVICE=${DEVICE:-cuda}
PY=${PY:-python}
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ARCH=(--hidden 128 --n-layers 3 --trunk-dims 192,160,128 --mlp-activation silu --mlp-layer-norm
      --embed-dim 32 --hist-d-model 96 --hist-nhead 4 --hist-n-layers 2 --hist-ffn-dim 192
      --hist-norm-first --hist-activation gelu --split-actor-critic-history --static-attention)
RUN=(--games-per-update 1024 --workers ${WORKERS:-4} --synchronous --batch-size 512 --optimizer-batch-size 2048
     --train-gt --gt-probe-every 200000 --gt-probe-games 40000
     --checkpoint-every ${CKPT_EVERY:-250000} --lr 3e-4
     --entropy-coef 0.03 --entropy-coef-end 0.01 --entropy-decay-games 2000000 --entropy-decay-start 0
     --no-gt-explore --gt-probe-practice 0)
MMD=(--mmd-alpha 0.03 --mmd-epsilon 0.2
     --mmd-window-updates 100 --mmd-min-round-games 200000
     --mmd-max-round-games 3000000 --mmd-eval-deals 16384 --mmd-eval-procs ${EVAL_PROCS:-4})
mkdir -p "$OUT/checkpoints"
for attempt in $(seq 1 50); do
  RESUME=()
  [ -f "$OUT/checkpoints/latest.pt" ] && RESUME=(--resume "$OUT/checkpoints/latest.pt")
  echo "=== launch attempt $attempt $(date -Is) ${RESUME[*]:-from scratch}"
  "$PY" -m training.train_history "${RESUME[@]}" \
    --checkpoint-dir "$OUT/checkpoints" --log-file "$OUT/train.csv" --games "$GAMES" \
    --device "$DEVICE" "${ARCH[@]}" "${RUN[@]}" "${MMD[@]}"
  code=$?
  echo "=== exit $code $(date -Is)"
  [ "$code" -eq 0 ] && break
  [ "$code" -eq 130 ] && break   # Ctrl-C: stop, do not resume
  sleep 60
done
