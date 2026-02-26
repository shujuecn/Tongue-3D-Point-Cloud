#!/usr/bin/env bash
set -euo pipefail

AE_CONFIG="${AE_CONFIG:-configs/autoencoder_4090_dense.yaml}"
IMG_CONFIG="${IMG_CONFIG:-configs/image2shape_4090_dense.yaml}"
SAMPLE_ID="${SAMPLE_ID:-03903.000052}"

mkdir -p runs/tmp runs/predictions runs/compare runs/renders
export MPLCONFIGDIR="${MPLCONFIGDIR:-runs/tmp/mplconfig}"
mkdir -p "$MPLCONFIGDIR"

if [[ ! -f "$AE_CONFIG" ]]; then
  echo "[error] AE config not found: $AE_CONFIG" >&2
  exit 1
fi
if [[ ! -f "$IMG_CONFIG" ]]; then
  echo "[error] IMG2SHAPE config not found: $IMG_CONFIG" >&2
  exit 1
fi

echo "[1/6] Train autoencoder with $AE_CONFIG"
python -m tongue3d.scripts.train_autoencoder "$AE_CONFIG"

readarray -t AE_INFO < <(
python - "$AE_CONFIG" <<'PY'
from pathlib import Path
import yaml
import sys
cfg_path = Path(sys.argv[1])
cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
out_root = Path(cfg['output_dir'])
latest_file = out_root / 'latest_run.txt'
latest = Path(latest_file.read_text(encoding='utf-8').strip()) if latest_file.exists() else out_root
best = out_root / 'best.pt'
if not best.exists():
    best = latest / 'best.pt'
print(str(out_root))
print(str(latest))
print(str(best))
PY
)
AE_ROOT="${AE_INFO[0]}"
AE_RUN="${AE_INFO[1]}"
AE_BEST="${AE_INFO[2]}"

if [[ ! -f "$AE_BEST" ]]; then
  echo "[error] AE best checkpoint not found: $AE_BEST" >&2
  exit 1
fi

IMG_EFFECTIVE_CFG="runs/tmp/image2shape_effective.yaml"
python - "$IMG_CONFIG" "$AE_BEST" "$IMG_EFFECTIVE_CFG" <<'PY'
from pathlib import Path
import sys
import yaml
img_cfg_path = Path(sys.argv[1])
ae_best = Path(sys.argv[2])
out_cfg_path = Path(sys.argv[3])
cfg = yaml.safe_load(img_cfg_path.read_text(encoding='utf-8'))
cfg['autoencoder_checkpoint'] = str(ae_best)
out_cfg_path.parent.mkdir(parents=True, exist_ok=True)
out_cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding='utf-8')
print(out_cfg_path)
PY

echo "[2/6] Train image2shape with $IMG_EFFECTIVE_CFG"
python -m tongue3d.scripts.train_image2shape "$IMG_EFFECTIVE_CFG"

readarray -t IMG_INFO < <(
python - "$IMG_EFFECTIVE_CFG" <<'PY'
from pathlib import Path
import yaml
import sys
cfg_path = Path(sys.argv[1])
cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
out_root = Path(cfg['output_dir'])
latest_file = out_root / 'latest_run.txt'
latest = Path(latest_file.read_text(encoding='utf-8').strip()) if latest_file.exists() else out_root
best = out_root / 'best.pt'
if not best.exists():
    best = latest / 'best.pt'
print(str(out_root))
print(str(latest))
print(str(best))
PY
)
IMG_ROOT="${IMG_INFO[0]}"
IMG_RUN="${IMG_INFO[1]}"
IMG_BEST="${IMG_INFO[2]}"

if [[ ! -f "$IMG_BEST" ]]; then
  echo "[error] image2shape best checkpoint not found: $IMG_BEST" >&2
  exit 1
fi

echo "[3/6] Evaluate val split"
python -m tongue3d.scripts.evaluate "$IMG_BEST" val "$IMG_RUN/splits.csv" "$IMG_RUN/eval_val.json"

PRED_PLY="runs/predictions/${SAMPLE_ID}_pred.ply"
GT_OBJ="TongueDB/meshes/${SAMPLE_ID}.obj"

echo "[4/6] Inference for sample_id=$SAMPLE_ID"
python -m tongue3d.scripts.infer_single "$IMG_BEST" "$SAMPLE_ID" "$PRED_PLY" "$IMG_RUN/splits.csv"

if [[ -f "$GT_OBJ" ]]; then
  echo "[5/6] Render GT/Pred comparison"
  python -m tongue3d.scripts.visualize_compare "$GT_OBJ" "$PRED_PLY" "runs/compare/${SAMPLE_ID}_compare.png" 8192
fi

echo "[6/6] Render paper-style blue splat"
python -m tongue3d.scripts.render_blue_splat "$PRED_PLY" "runs/renders/${SAMPLE_ID}_pred_blue_splat.png" 12000 1.0

if [[ -f "$GT_OBJ" ]]; then
  python -m tongue3d.scripts.render_blue_splat "$GT_OBJ" "runs/renders/${SAMPLE_ID}_gt_blue_splat.png" 12000 1.0
fi

echo "[done]"
echo "  AE run:  $AE_RUN"
echo "  IMG run: $IMG_RUN"
echo "  Pred:    $PRED_PLY"
echo "  Compare: runs/compare/${SAMPLE_ID}_compare.png"
echo "  Render:  runs/renders/${SAMPLE_ID}_pred_blue_splat.png"
echo "  TensorBoard: tensorboard --logdir $IMG_RUN/tensorboard"
