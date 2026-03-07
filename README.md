# Tongue 2D->3D 重建复现（PyTorch）

本项目用于从单张舌象图像重建 3D 舌体点云，核心是两阶段训练：

1. 阶段一：`Point AutoEncoder (AE)` 学习 3D 舌体潜空间与解码器。
2. 阶段二：`Image2Shape` 学习 `2D图像 -> latent -> 3D点云`。

在当前版本中，已接入你提供的两类增强数据：

- `TongueDB/masks`（1828 对监督样本的舌体分割）
- 5000+ 真实世界彩色图/分割图（无 3D 真值）

---

## 1. 复现程度（当前状态）

按“论文主干+可运行工程”衡量，当前复现大约 **80% 左右**。

已完成：
- 两阶段主干训练与推理
- mask 感知训练（舌体 ROI）
- 真实世界一致性训练（无 3D）
- 训练日志、评估、可视化、单图推理
- 5000+ 数据二进制缓存加速（`in_the_wild_cache.npz`）
- 二阶段按论文风格的“先监督、后弱无监督”训练节奏
- 5000+ 变尺寸图像保比例预处理（`letterbox`，避免几何拉伸）

未完全复现：
- 论文 TongueGAN 的完整生成器-判别器链路
- 论文中与头部参数化/身份相关的完整后续链路
- 你未提供监督信息对应的模块（例如关键点或参数化身份标注）

结论：当前是可稳定训练和迭代优化的工程复现版本，不是论文全链路 1:1 复刻。

---

## 2. 目录与数据约定

```text
TongueDB/
  images/*.png                  # 监督样本 2D 图
  meshes/*.obj                  # 监督样本 3D 网格
  masks/*.png                   # 监督样本 mask，0/255，与 images 同名
  in_the_wild_pairs.csv         # 5000+ 真实世界配对清单（脚本生成）
  in_the_wild_cache.npz         # 5000+ 二进制缓存（脚本生成）

configs/
  autoencoder_4090_dense.yaml
  image2shape_4090_dense.yaml

train.sh                        # 训练专用入口（推荐）
run.sh                          # 全功能入口（eval/infer/visualize 等）
```

监督样本按文件名 stem 对齐，例如：
- `TongueDB/images/03903.000052.png`
- `TongueDB/meshes/03903.000052.obj`
- `TongueDB/masks/03903.000052.png`

---

## 3. 环境

```bash
pip install -r requirements.txt
```

建议在 WSL 中运行，并优先使用 `train.sh`。

---

## 4. 推荐运行流程（当前最佳实践）

### 4.1 生成 5000+ 配对清单（只需一次）

输入路径必须是 WSL 路径风格：`/mnt/<盘符>/...`

```bash
./train.sh prepare-wild \
  /mnt/f/数据集/Tongue-250106-5375-ZDWY（241228-ZDWY的修订版）/1-Tongue-ColorCorrected-250106-5375 \
  /mnt/f/数据集/Tongue-250106-5375-ZDWY（241228-ZDWY的修订版）/2-Tongue-Segmented-250106-5375 \
  TongueDB/in_the_wild_pairs.csv
```

### 4.2 生成二进制缓存（只需一次，强烈推荐）

```bash
./train.sh cache-wild TongueDB/in_the_wild_pairs.csv TongueDB/in_the_wild_cache.npz 224 1 letterbox 16
```

参数说明：
- `224`：缓存图像尺寸（需与配置一致）
- `1`：对彩色图应用 segmented mask 背景抑制
- `letterbox`：保持长宽比后再补边到正方形（推荐，适合变尺寸数据）
- `16`：分割阈值（抑制 JPG 黑底噪声）

如果你改了 `resize_mode` / `segmented_mask_threshold` / `image_size`，必须重建缓存。

### 4.3 训练阶段一 AE

```bash
./train.sh ae configs/autoencoder_4090_dense.yaml
```

### 4.4 训练阶段二 Image2Shape（推荐命令）

```bash
TONGUE3D_ALLOW_WSL_WORKERS=1 TONGUE3D_NUM_WORKERS=4 \
./train.sh img configs/image2shape_4090_dense.yaml runs/ae_4090_dense/best.pt
```

说明：
- `TONGUE3D_ALLOW_WSL_WORKERS=1`：允许 WSL 多进程 DataLoader
- `TONGUE3D_NUM_WORKERS=4`：覆盖配置里的 worker 数

如果不稳定，先降到 `2`，再不稳定回退到 `0`。

---

## 5. 两阶段模型架构（详细）

### 5.1 阶段一：Point AutoEncoder

代码：`tongue3d/models/point_autoencoder.py`

编码器 `PointNetEncoder`：
- 输入：`B x N x 3`
- `Conv1d(3->64, k=1) + BN + ReLU`
- `Conv1d(64->128, k=1) + BN + ReLU`
- `Conv1d(128->256, k=1) + BN + ReLU`
- `Conv1d(256->512, k=1) + BN + ReLU`
- 全局最大池化（点维度）
- `Linear(512->512) + ReLU + Dropout`
- `Linear(512->latent_dim)`

解码器 `PointDecoder`：
- 输入：`B x latent_dim`
- `Linear(latent_dim->hidden_dim) + ReLU + Dropout`
- `Linear(hidden_dim->hidden_dim) + ReLU + Dropout`
- `Linear(hidden_dim->num_points*6)`
- reshape 为 `B x num_points x 6`
- 前 3 维 `tanh` 作为坐标，后 3 维 `L2 normalize` 作为法向

阶段一作用：
- 学到稳定的舌体 3D 潜空间
- 产出可复用 decoder（给阶段二）

### 5.2 阶段二：Image2Shape

代码：`tongue3d/models/image_encoder.py` + `image_to_shape.py`

图像编码器 `TongueImageEncoder`：
- 主干：`ResNet50`（默认 ImageNet 预训练）
- 去掉原分类头，接 MLP 头：
  - `Linear(in_features->1024) + ReLU + Dropout`
  - `Linear(1024->latent_dim)`

latent 映射器 `LatentMapper`：
- `Linear(latent_dim->latent_dim) + ReLU + Dropout + Linear(latent_dim->latent_dim)`

解码器：
- 直接复用阶段一 AE 的 `PointDecoder`

正向流程：
1. `image -> image_encoder -> latent_raw`
2. `latent_raw -> mapper -> latent`
3. `latent -> decoder -> points + normals`

### 5.3 真实世界一致性分支（5000+ 无3D）

代码：`tongue3d/scripts/train_image2shape.py`

对每个 in-the-wild 样本，取两种输入：
- 彩色图（可含背景抑制）
- 分割图（黑底舌体）

输入预处理：
- 保比例缩放 + 补边（`letterbox`），避免变尺寸样本被强行拉伸
- 分割阈值默认 `16`，减少压缩噪声误检

一致性损失：
- 仅计算 latent，不走 decoder（已优化提速）
- `L_consistency = MSE(latent(color), latent(segmented))`

总损失中叠加（论文风格课程训练）：
- 前期仅监督训练（warmup）
- 从 `start_epoch` 开始按 `consistency_ramp_epochs` 线性爬升权重
- 每个 epoch 只抽样有限步数 in-the-wild（`max_steps_per_epoch`）
- `L_total = L_supervised + w(epoch) * L_consistency`

---

## 6. 关键训练参数（当前配置）

### 6.1 AE：`configs/autoencoder_4090_dense.yaml`

- `epochs: 220`
- `batch_size: 8`
- `grad_accum_steps: 2`
- `dataset.num_points: 8192`
- `model.latent_dim: 256`
- `model.decoder_hidden_dim: 1536`
- `optimizer.lr: 1.5e-4`
- `loss`：
  - `chamfer: 1.0`
  - `normal: 0.1`
  - `laplacian: 0.07`
  - `edge: 0.03`
  - `repulsion: 0.02`

### 6.2 Image2Shape：`configs/image2shape_4090_dense.yaml`

基础参数：
- `epochs: 170`
- `batch_size: 8`
- `grad_accum_steps: 2`
- `dataset.num_points: 8192`
- `model.latent_dim: 256`
- `freeze_decoder: true`（更贴近论文二阶段“固定解码器”）
- `decoder_lr_scale: 0.2`
- `optimizer.lr: 8e-5`

监督损失：
- `chamfer: 1.2`
- `normal: 0.08`
- `laplacian: 0.08`
- `edge: 0.03`
- `repulsion: 0.02`
- `latent: 1.2`

mask 感知参数：
- `dataset.use_mask: true`
- `dataset.mask_crop: true`
- `dataset.mask_background_zero: true`
- `dataset.mask_threshold: 127`
- `dataset.mask_margin_ratio: 0.08`

in-the-wild 参数：
- `in_the_wild.enabled: true`
- `in_the_wild.use_binary_cache: true`
- `in_the_wild.binary_cache_path: TongueDB/in_the_wild_cache.npz`
- `in_the_wild.batch_size: 8`
- `in_the_wild.augment: false`
- `in_the_wild.start_epoch: 20`
- `in_the_wild.consistency_ramp_epochs: 20`
- `in_the_wild.max_steps_per_epoch: 32`
- `in_the_wild.resize_mode: letterbox`
- `in_the_wild.segmented_mask_threshold: 16`
- `loss.in_the_wild_consistency: 0.08`

---

## 7. 低 GPU 利用率排查与调优

你当前日志里的优化路径是正确的。

建议顺序：
1. 先生成 `in_the_wild_cache.npz`，避免每 step 大量随机读图。
2. 开启 WSL 多 worker：
   - `TONGUE3D_ALLOW_WSL_WORKERS=1`
   - `TONGUE3D_NUM_WORKERS=4`
3. 观察启动日志：
   - `[loader] train_workers=4 ...`
   - `[in_the_wild] ... workers=4`

如果遇到不稳定：
- 先把 `TONGUE3D_NUM_WORKERS` 降到 `2`
- 仍不稳就回退 `0`

---

## 8. 训练输出与检查点

输出目录：`runs/<experiment_name>/<timestamp_run>/`

典型文件：
- `config.json`
- `normalization.json`
- `splits.csv`
- `metrics.csv`
- `tensorboard/`
- `best.pt`
- `last.pt`

---

## 9. 评估与推理（非训练）

```bash
# 评估
./run.sh eval runs/img2shape_4090_dense/best.pt

# 单图推理
./run.sh infer runs/img2shape_4090_dense/best.pt TongueDB/images/03903.000052.png

# 可视化
./run.sh visualize TongueDB/meshes/03903.000052.obj runs/predictions/03903.000052_pred.ply
./run.sh render runs/predictions/03903.000052_pred.ply
```

说明：若 checkpoint 配置启用了 `dataset.use_mask=true`，且存在同名 mask  
（`TongueDB/masks/<sample_id>.png`），推理会自动复用训练同款 mask 预处理。

---

## 10. 能力边界与下一步

当前版本已能稳定训练并利用你新增数据进行论文风格适配，但仍属于工程复现路线。

如果后续目标是进一步逼近论文完整链路，下一步建议优先做：
1. TongueGAN 完整生成器-判别器训练分支。
2. “嘴唇聚点率”等专门诊断指标接入验证流程。
3. 结构化 ablation（mask 开关、consistency 权重、cache/worker 对速度和质量影响）。
