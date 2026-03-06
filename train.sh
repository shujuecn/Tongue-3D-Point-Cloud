#!/usr/bin/env bash
set -euo pipefail

AE_CONFIG_DEFAULT="${AE_CONFIG:-configs/autoencoder_4090_dense.yaml}"
IMG_CONFIG_DEFAULT="${IMG_CONFIG:-configs/image2shape_4090_dense.yaml}"
WILD_MANIFEST_DEFAULT="${WILD_MANIFEST:-TongueDB/in_the_wild_pairs.csv}"

usage() {
  cat <<'USAGE'
Usage:
  ./train.sh prepare-wild <color_dir> <segmented_dir> [output_csv]
  ./train.sh ae [ae_config]
  ./train.sh img [img_config] [ae_checkpoint]
  ./train.sh full [ae_config] [img_config] [ae_checkpoint]

Notes:
  - This script is training-only (no eval/infer/visualize).
  - prepare-wild requires WSL mount paths like /mnt/f/... (Windows-style paths are rejected).
USAGE
}

require_file() {
  local target="$1"
  if [[ ! -f "$target" ]]; then
    echo "[error] file not found: $target" >&2
    exit 1
  fi
}

find_ae_best_checkpoint() {
  local ae_config="$1"
  python - "$ae_config" <<'PY'
from pathlib import Path
import sys
import yaml

cfg_path = Path(sys.argv[1])
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
root = Path(cfg["output_dir"])
best = root / "best.pt"
if best.exists():
    print(best)
    raise SystemExit(0)

latest_file = root / "latest_run.txt"
if latest_file.exists():
    run_dir = Path(latest_file.read_text(encoding="utf-8").strip())
    run_best = run_dir / "best.pt"
    if run_best.exists():
        print(run_best)
        raise SystemExit(0)

print("")
PY
}

cmd_prepare_wild() {
  if [[ $# -lt 2 ]]; then
    echo "[error] prepare-wild requires: <color_dir> <segmented_dir> [output_csv]" >&2
    usage
    exit 1
  fi
  local color_dir="$1"
  local segmented_dir="$2"
  local output_csv="${3:-$WILD_MANIFEST_DEFAULT}"

  echo "[prepare-wild] color=$color_dir segmented=$segmented_dir output=$output_csv"
  python -m tongue3d.scripts.prepare_in_the_wild_pairs "$color_dir" "$segmented_dir" "$output_csv"
}

cmd_train_ae() {
  local ae_config="${1:-$AE_CONFIG_DEFAULT}"
  require_file "$ae_config"
  echo "[train-ae] config=$ae_config"
  python -m tongue3d.scripts.train_autoencoder "$ae_config"
}

cmd_train_img() {
  local img_config="${1:-$IMG_CONFIG_DEFAULT}"
  local ae_ckpt="${2:-}"
  require_file "$img_config"

  if [[ -n "$ae_ckpt" ]]; then
    require_file "$ae_ckpt"
    echo "[train-img] config=$img_config ae_checkpoint=$ae_ckpt"
    ./run.sh train-img "$img_config" "$ae_ckpt"
  else
    echo "[train-img] config=$img_config"
    python -m tongue3d.scripts.train_image2shape "$img_config"
  fi
}

cmd_train_full() {
  local ae_config="${1:-$AE_CONFIG_DEFAULT}"
  local img_config="${2:-$IMG_CONFIG_DEFAULT}"
  local ae_ckpt="${3:-}"

  cmd_train_ae "$ae_config"

  if [[ -z "$ae_ckpt" ]]; then
    ae_ckpt="$(find_ae_best_checkpoint "$ae_config")"
  fi
  if [[ -z "$ae_ckpt" ]]; then
    echo "[error] unable to find AE best checkpoint after training" >&2
    exit 1
  fi
  require_file "$ae_ckpt"
  cmd_train_img "$img_config" "$ae_ckpt"
}

main() {
  local command="${1:-help}"
  if [[ $# -gt 0 ]]; then
    shift
  fi

  case "$command" in
    prepare-wild)
      cmd_prepare_wild "$@"
      ;;
    ae)
      cmd_train_ae "$@"
      ;;
    img)
      cmd_train_img "$@"
      ;;
    full)
      cmd_train_full "$@"
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      echo "[error] unknown command: $command" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"

