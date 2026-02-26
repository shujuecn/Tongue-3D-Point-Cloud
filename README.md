# Tongue 2D -> 3D Reconstruction (PyTorch)

This repository contains a practical training pipeline for **single-image tongue 3D reconstruction** based on the public `TongueDB` data and the method idea from paper `2106.12302v1`.

Because the paper's original release does not include UHM rigged PCA assets or official training code, this implementation uses a reproducible substitute while keeping the same core training logic:

1. Train a point-cloud autoencoder to learn a 3D latent (`y in R^256`).
2. Train an image encoder to regress that latent.
3. Decode latent back to 3D tongue point cloud.

## Implemented stack

- `PyTorch` for models/training
- `pydantic` for all configuration objects
- `tqdm` for progress display
- **No `argparse` is used**

## Project layout

- `tongue3d/data`: dataset pairing, split, mesh/image dataset
- `tongue3d/models`: point AE, image encoder, image-to-shape model
- `tongue3d/losses`: Chamfer, normal alignment, smoothness regularizers
- `tongue3d/scripts/train_autoencoder.py`: stage-1 training
- `tongue3d/scripts/train_image2shape.py`: stage-2 training
- `tongue3d/scripts/infer_single.py`: single-image inference to `.ply`
- `configs/autoencoder.yaml`
- `configs/image2shape.yaml`

## Data expectation

Expected paths:

- `TongueDB/images/*.png`
- `TongueDB/meshes/*.obj`

The loader pairs samples by filename stem.

## Environment

```bash
pip install -r requirements.txt
```

## Training

### Stage 1: point-cloud autoencoder

```bash
python -m tongue3d.scripts.train_autoencoder
```

Optional config override without argparse:

```bash
TONGUE3D_CONFIG=configs/autoencoder.yaml python -m tongue3d.scripts.train_autoencoder
# or
python -m tongue3d.scripts.train_autoencoder configs/autoencoder.yaml
```

For RTX 4090 dense output (8192 points):

```bash
python -m tongue3d.scripts.train_autoencoder configs/autoencoder_4090_dense.yaml
```

Outputs under `runs/ae_baseline/`:

- `best.pt`
- `last.pt`
- `normalization.json`
- `config.json`

### Stage 2: image -> 3D

Edit `configs/image2shape.yaml` `autoencoder_checkpoint` first, then run:

```bash
python -m tongue3d.scripts.train_image2shape
```

For RTX 4090 dense output (8192 points):

```bash
python -m tongue3d.scripts.train_image2shape configs/image2shape_4090_dense.yaml
```

Outputs under `runs/img2shape_baseline/`.

## Inference

```bash
python -m tongue3d.scripts.infer_single runs/img2shape_baseline/best.pt TongueDB/images/04017.000055.png runs/predictions/04017.000055.ply
```

## Notes on paper alignment

- Preserved:
  - two-stage learning (`AE latent -> image regression -> 3D reconstruction`)
  - latent supervision (`L_y` equivalent)
  - geometric losses (Chamfer + normal + smoothness regularizers)
- Replaced (due to missing private assets):
  - UHM rigged PCA decoder -> neural point decoder
  - collision term around mouth landmarks -> omitted
  - TongueGAN in-the-wild synthesis -> not included in this baseline

## 4090 (24GB) suggestions

- `num_points: 2048`
- `batch_size: 24` (AMP on)
- `num_workers: 8`
- keep `preload_meshes: true` (sufficient with 64GB RAM)

For denser reconstruction close to your mesh vertex density:

- use `configs/autoencoder_4090_dense.yaml` and `configs/image2shape_4090_dense.yaml`
- output is `8192` points per sample
- Chamfer uses chunking (`chamfer_chunk_size: 2048`) to control memory
