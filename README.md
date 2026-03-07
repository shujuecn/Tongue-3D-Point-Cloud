# Tongue 2D->3D Baseline 分支说明

本分支是 `baseline` 配置分支，目标是保留与 `main` 一致的代码结构与脚本入口，仅回退到最初的纯监督训练设置，便于做稳定对比。

## 1. 这个分支做了什么

- 保留 `main` 的代码结构（`run.sh`、`train.sh`、训练脚本、数据类都不删）。
- 将二阶段配置回退到纯监督：
  - `freeze_decoder: false`
  - `dataset.use_mask: false`
  - `dataset.mask_crop: false`
  - `in_the_wild.enabled: false`
- 不使用 `TongueDB/masks`、不使用 5000+ in-the-wild 数据参与训练。

核心配置文件：
- `configs/autoencoder_4090_dense.yaml`
- `configs/image2shape_4090_dense.yaml`

## 2. 数据要求（baseline）

只需要监督对：

```text
TongueDB/
  images/*.png
  meshes/*.obj
```

`masks/`、`in_the_wild_pairs.csv`、`in_the_wild_cache.npz` 在本分支训练流程中不是必需。

## 3. 推荐训练流程

### 3.1 阶段一：训练 AE

```bash
./train.sh ae configs/autoencoder_4090_dense.yaml
```

### 3.2 阶段二：训练 Image2Shape（纯监督）

```bash
./train.sh img configs/image2shape_4090_dense.yaml runs/ae_4090_dense/best.pt
```

如果 WSL 下需要提速：

```bash
TONGUE3D_ALLOW_WSL_WORKERS=1 TONGUE3D_NUM_WORKERS=4 \
./train.sh img configs/image2shape_4090_dense.yaml runs/ae_4090_dense/best.pt
```

## 4. 评估与推理

```bash
./run.sh eval runs/img2shape_4090_dense/best.pt
./run.sh infer runs/img2shape_4090_dense/best.pt TongueDB/images/03903.000052.png
```

## 5. 与 main 的对比

`baseline` 分支：
- 纯监督（不使用 mask，不使用 in-the-wild）
- 用于建立可重复的基线结果

`main` 分支：
- 含 mask 监督与 in-the-wild 一致性训练
- 用于增强泛化和域适配实验

查看差异：

```bash
git diff --stat main..codex/baseline
git diff main..codex/baseline -- configs/image2shape_4090_dense.yaml README.md
```

## 6. 分支切换与互转

切回 baseline：

```bash
git checkout codex/baseline
```

切回 main：

```bash
git checkout main
```

如果你想把 baseline 配置临时用于 main 训练，不改代码只改配置：

```bash
git show codex/baseline:configs/image2shape_4090_dense.yaml > configs/image2shape_4090_dense.yaml
```

如果你想恢复 main 原配置：

```bash
git checkout main -- configs/image2shape_4090_dense.yaml
```

## 7. 说明

本分支不合并进 `main`，仅用于基线对照实验与参数回归测试。
