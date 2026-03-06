#!/usr/bin/env bash
set -euo pipefail

AE_CONFIG_DEFAULT="${AE_CONFIG:-configs/autoencoder_4090_dense.yaml}"
IMG_CONFIG_DEFAULT="${IMG_CONFIG:-configs/image2shape_4090_dense.yaml}"
TMP_DIR="${TMP_DIR:-runs/tmp}"

usage() {
  cat <<'USAGE'
Usage:
  ./run.sh train-ae [ae_config]
  ./run.sh train-img [img_config] [ae_checkpoint]
  ./run.sh prepare-wild <color_dir> <segmented_dir> [output_csv]
  ./run.sh eval <img2shape_checkpoint> [split=val] [split_csv] [output_json]
  ./run.sh infer <img2shape_checkpoint> <image_path> [output_ply]
  ./run.sh visualize <gt_obj> <pred_ply> [output_png] [max_points=8192]
  ./run.sh render <point_cloud.(ply|obj)> [output_png] [max_points=12000] [size_scale=1.0]

Notes:
  - infer is image-path based by design: no split CSV is required.
  - train-img uses config's autoencoder_checkpoint by default.
  - train-img can override AE checkpoint via the 2nd optional arg.
  - prepare-wild writes a matched manifest CSV under TongueDB by default.
  - prepare-wild requires /mnt/<drive>/... paths in WSL.
USAGE
}

require_file() {
  local target="$1"
  if [[ ! -f "$target" ]]; then
    echo "[error] file not found: $target" >&2
    exit 1
  fi
}

# 仅在需要 matplotlib 的命令中初始化缓存目录，避免 infer 时创建无关空目录。
setup_mpl_runtime() {
  local mpl_dir="${MPLCONFIGDIR:-$TMP_DIR/mplconfig}"
  export MPLCONFIGDIR="$mpl_dir"
  mkdir -p "$mpl_dir"
}

# 仅在需要写临时配置时创建 TMP_DIR。
ensure_tmp_dir() {
  mkdir -p "$TMP_DIR"
}

build_image2shape_config_with_ae() {
  local img_config="$1"
  local ae_ckpt="$2"
  ensure_tmp_dir
  local out_cfg="$TMP_DIR/image2shape_effective_$(date +%Y%m%d_%H%M%S).yaml"

  python - "$img_config" "$ae_ckpt" "$out_cfg" <<'PY'
from pathlib import Path
import sys
import yaml

img_cfg_path = Path(sys.argv[1])
ae_ckpt = Path(sys.argv[2])
out_cfg_path = Path(sys.argv[3])

cfg = yaml.safe_load(img_cfg_path.read_text(encoding='utf-8'))
cfg['autoencoder_checkpoint'] = str(ae_ckpt)
out_cfg_path.parent.mkdir(parents=True, exist_ok=True)
out_cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding='utf-8')
print(out_cfg_path)
PY
}

cmd_train_ae() {
  local ae_config="${1:-$AE_CONFIG_DEFAULT}"
  require_file "$ae_config"
  setup_mpl_runtime

  echo "[train-ae] config=$ae_config"
  python -m tongue3d.scripts.train_autoencoder "$ae_config"
}

cmd_train_img() {
  local img_config="${1:-$IMG_CONFIG_DEFAULT}"
  local ae_ckpt="${2:-}"
  require_file "$img_config"
  setup_mpl_runtime

  if [[ -n "$ae_ckpt" ]]; then
    require_file "$ae_ckpt"
    img_config="$(build_image2shape_config_with_ae "$img_config" "$ae_ckpt")"
    echo "[train-img] override autoencoder_checkpoint=$ae_ckpt"
    echo "[train-img] effective_config=$img_config"
  fi

  echo "[train-img] config=$img_config"
  python -m tongue3d.scripts.train_image2shape "$img_config"
}

cmd_eval() {
  if [[ $# -lt 1 ]]; then
    echo "[error] eval requires: <img2shape_checkpoint> [split] [split_csv] [output_json]" >&2
    usage
    exit 1
  fi

  local ckpt="$1"
  local split="${2:-val}"
  local third="${3:-}"
  local fourth="${4:-}"
  local split_csv=""
  local output_json=""
  require_file "$ckpt"

  # 兼容：./run.sh eval ckpt val out.json（省略 split_csv）。
  if [[ -n "$third" && -z "$fourth" && "$third" == *.json ]]; then
    output_json="$third"
  else
    split_csv="$third"
    output_json="$fourth"
  fi

  if [[ -n "$split_csv" ]]; then
    require_file "$split_csv"
  fi

  local cmd=(python -m tongue3d.scripts.evaluate "$ckpt" "$split")
  if [[ -n "$split_csv" ]]; then
    cmd+=("$split_csv")
  elif [[ -n "$output_json" ]]; then
    cmd+=("")
  fi
  if [[ -n "$output_json" ]]; then
    cmd+=("$output_json")
  fi

  echo "[eval] checkpoint=$ckpt split=$split"
  "${cmd[@]}"
}

cmd_prepare_wild() {
  if [[ $# -lt 2 ]]; then
    echo "[error] prepare-wild requires: <color_dir> <segmented_dir> [output_csv]" >&2
    usage
    exit 1
  fi
  local color_dir="$1"
  local segmented_dir="$2"
  local output_csv="${3:-TongueDB/in_the_wild_pairs.csv}"
  echo "[prepare-wild] color=$color_dir segmented=$segmented_dir output=$output_csv"
  python -m tongue3d.scripts.prepare_in_the_wild_pairs "$color_dir" "$segmented_dir" "$output_csv"
}

cmd_infer() {
  if [[ $# -lt 2 ]]; then
    echo "[error] infer requires: <img2shape_checkpoint> <image_path> [output_ply]" >&2
    usage
    exit 1
  fi

  local ckpt="$1"
  local image_path="$2"
  local output_ply="${3:-}"
  require_file "$ckpt"
  require_file "$image_path"

  local cmd=(python -m tongue3d.scripts.infer_single "$ckpt" "$image_path")
  if [[ -n "$output_ply" ]]; then
    cmd+=("$output_ply")
  fi

  echo "[infer] checkpoint=$ckpt image=$image_path"
  "${cmd[@]}"
}

cmd_visualize() {
  if [[ $# -lt 2 ]]; then
    echo "[error] visualize requires: <gt_obj> <pred_ply> [output_png] [max_points]" >&2
    usage
    exit 1
  fi

  local gt_obj="$1"
  local pred_ply="$2"
  local output_png="${3:-runs/compare/compare.png}"
  local max_points="${4:-8192}"
  require_file "$gt_obj"
  require_file "$pred_ply"
  setup_mpl_runtime

  echo "[visualize] gt=$gt_obj pred=$pred_ply"
  python -m tongue3d.scripts.visualize_compare "$gt_obj" "$pred_ply" "$output_png" "$max_points"
}

cmd_render() {
  if [[ $# -lt 1 ]]; then
    echo "[error] render requires: <point_cloud.(ply|obj)> [output_png] [max_points] [size_scale]" >&2
    usage
    exit 1
  fi

  local point_cloud="$1"
  local output_png="${2:-runs/renders/render_blue_splat.png}"
  local max_points="${3:-12000}"
  local size_scale="${4:-1.0}"
  require_file "$point_cloud"
  setup_mpl_runtime

  echo "[render] input=$point_cloud"
  python -m tongue3d.scripts.render_blue_splat "$point_cloud" "$output_png" "$max_points" "$size_scale"
}

main() {
  local command="${1:-help}"
  if [[ $# -gt 0 ]]; then
    shift
  fi

  case "$command" in
    train-ae)
      cmd_train_ae "$@"
      ;;
    train-img)
      cmd_train_img "$@"
      ;;
    eval)
      cmd_eval "$@"
      ;;
    prepare-wild)
      cmd_prepare_wild "$@"
      ;;
    infer)
      cmd_infer "$@"
      ;;
    visualize)
      cmd_visualize "$@"
      ;;
    render)
      cmd_render "$@"
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
