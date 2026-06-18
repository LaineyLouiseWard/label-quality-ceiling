#!/bin/bash
# ============================================================================
# Multi-seed campaign driver — ONE resumable command for the whole campaign.
#
#   bash run_campaign.sh            # run / resume the full 5-seed campaign
#   DRYRUN=1 bash run_campaign.sh   # print the exact plan, run nothing
#
# You do NOT need to remember which seed or stage you were on. Re-run this after
# ANY interruption (Ctrl-C, reboot, CUDA crash) and it picks up automatically:
#   - finished seeds are skipped (tracked in .campaign/seed_<N>.done),
#   - the in-progress seed resumes mid-stage from its last.ckpt (RESUME=1),
#   - remaining seeds then run, and finally the cross-seed aggregate is built.
#
# Run it inside tmux so it survives the terminal closing:
#   conda activate ClassImbalance
#   tmux new -s seeds          # detach: Ctrl-b then d   |   reattach: tmux attach -t seeds
#   bash run_campaign.sh
#
# To force ONE seed to re-run from scratch: delete .campaign/seed_<N>.done (and,
# for the root seed, .campaign/wiped) and its model_weights, then re-run.
# ============================================================================
set -euo pipefail

SEEDS=(42 43 44 45 46)            # LOCKED campaign seed set
ROOT_SEED=42                      # the seed that lives in this repo (others = worktrees)
TEACHER=pretrain_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # this repo = root seed
STATE="$ROOT/.campaign"
mkdir -p "$STATE"

# Pin worktrees to the root's current commit, DETACHED (not a branch): a branch can
# only be checked out in one worktree, and the seeds only ever write gitignored
# outputs, so they never need their own branch. Keep the code fixed during the run.
CAMPAIGN_COMMIT="$(cd "$ROOT" && git rev-parse HEAD)"

# Fail fast if the conda env is not active (else the first seed dies hours in).
if ! python -c "import torch" >/dev/null 2>&1; then
  echo "ERROR: active 'python' has no PyTorch — run 'conda activate ClassImbalance' first."
  exit 1
fi

run() {   # echo the command; execute it unless DRYRUN=1
  echo "+ $*"
  [ "${DRYRUN:-0}" = "1" ] || "$@"
}

for SEED in "${SEEDS[@]}"; do
  DONE="$STATE/seed_$SEED.done"
  if [ -f "$DONE" ]; then
    echo "================= seed $SEED: already complete, skipping ================="
    continue
  fi
  echo "========================= seed $SEED ========================="

  if [ "$SEED" = "$ROOT_SEED" ]; then
    DIR="$ROOT"
    # Clear orphaned old-named checkpoints ONCE, before the very first root run.
    if [ ! -f "$STATE/wiped" ]; then
      run rm -rf "$ROOT"/model_weights/biodiversity/*
      run touch "$STATE/wiped"
    fi
  else
    DIR="$(dirname "$ROOT")/$(basename "$ROOT")_seed$SEED"
    # Create the worktree + shared symlinks on first touch (all idempotent).
    [ -d "$DIR" ] || run git -C "$ROOT" worktree add --detach "$DIR" "$CAMPAIGN_COMMIT"
    run ln -sfn "$ROOT/data" "$DIR/data"
    run mkdir -p "$DIR/pretrain_weights"
    run ln -sfn "$ROOT/$TEACHER" "$DIR/$TEACHER"
  fi

  # The resumable per-seed run: Stage 1 .. test eval (qualitative figures/analyses
  # are a separate once-off post-campaign step, not run per seed).
  run bash -c "cd '$DIR' && RESUME=1 SEED=$SEED HF_HUB_OFFLINE=1 RUN_NULL_CONTROLS=1 \
      bash RUNBOOK.sh --from B1 --to C2"

  run touch "$DONE"
  echo "========================= seed $SEED: complete ========================="
done

echo "========================= aggregating all seeds ========================="
run bash -c "cd '$ROOT' && PYTHONPATH=. python scripts/analysis/aggregate_seeds.py --strict"
echo "Campaign complete → analysis/seed_aggregate/ (summary.json, report.md, ablation_*.tex, figure10_iou_*.csv)"
