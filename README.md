# Tongue 2D->3D 重建复现（PyTorch）

本仓库是对参考 CVPR 工作的工程化复现，目标是从单张舌象图像重建 3D 舌体点云。

## 1. 当前复现程度（重点）

按可运行功能和论文主干对齐程度评估，当前复现大约在 **75%~85%**：

- 已完成
  - 两阶段主干：`Point AE -> Image2Shape`
  - 2D-3D 监督训练与验证
  - 单张图像推理、评估、可视化
  - mask 感知训练（`TongueDB/masks`）
  - 5000+ 真实世界图像的一致性训练（无 3D 真值）
  - 实验记录（`metrics.csv`/`tensorboard`/`splits.csv`）

- 未完全复现（实事求是）
  - 论文 TongueGAN 的完整生成器-判别器链路
  - 论文中涉及的头部/身份参数化重建链路（如 UHM 相关流程）
  - 你未提供的监督信息（例如关键点、3DMM 参数）

结论：本项目目前是“可训练、可推理、可扩展”的强工程复现版本，但不是论文全链路 1:1 复刻。

## 2. 项目结构

```text
TongueDB/
  images/*.png            # 2D 原图
  meshes/*.obj            # 3D 网格（监督训练）
  masks/*.png             # 0/255 舌体 mask（与 images 同名）
  in_the_wild_pairs.csv   # 5000+ 真实世界配对清单（由脚本生成）

configs/
  autoencoder_4090_dense.yaml
  image2shape_4090_dense.yaml

run.sh                    # 全功能入口（train/eval/infer/visualize/render/prepare-wild）
train.sh                  # 仅训练入口（你要求新增）

tongue3d/
  scripts/train_autoencoder.py
  scripts/train_image2shape.py
  scripts/prepare_in_the_wild_pairs.py
```

## 3. 环境准备

```bash
pip install -r requirements.txt
```

建议：WSL 下 `num_workers: 0`（已在默认配置中设置），可避免 DataLoader 相关崩溃。

## 4. 数据准备

### 4.1 监督数据（1828 对）

要求同名配对：
- `TongueDB/images/<sample_id>.png`
- `TongueDB/meshes/<sample_id>.obj`
- `TongueDB/masks/<sample_id>.png`

其中 `masks` 为 0/255 二值图（255=舌体）。

### 4.2 真实世界 5000+ 数据

你提供的数据用于域泛化一致性训练，输入必须是 **WSL 路径**（`/mnt/f/...`）。

先生成 manifest：

```bash
./train.sh prepare-wild \
  /mnt/f/数据集/Tongue-250106-5375-ZDWY（241228-ZDWY的修订版）/1-Tongue-ColorCorrected-250106-5375 \
  /mnt/f/数据集/Tongue-250106-5375-ZDWY（241228-ZDWY的修订版）/2-Tongue-Segmented-250106-5375 \
  TongueDB/in_the_wild_pairs.csv
```

注意：不再支持传 Windows 盘符路径（如 `F:\...`）。

## 5. 训练流程（推荐）

### 5.1 一键训练主流程（仅训练）

```bash
./train.sh full configs/autoencoder_4090_dense.yaml configs/image2shape_4090_dense.yaml
```

该命令会：
1. 训练阶段一 AE。
2. 自动定位 AE 的 `best.pt`。
3. 训练阶段二 Image2Shape。

### 5.2 分阶段训练

```bash
# 阶段一：AE
./train.sh ae configs/autoencoder_4090_dense.yaml

# 阶段二：Image2Shape（显式传入 AE 权重）
./train.sh img configs/image2shape_4090_dense.yaml runs/ae_4090_dense/best.pt
```

## 6. 训练方法说明（当前实现）

### 阶段一：Point AE

- 输入：3D 网格采样点云
- 输出：重建点云
- 作用：学习稳定的 3D 舌体形状潜空间和解码器

### 阶段二：Image2Shape

- 输入：2D 舌象图
- 输出：3D 点云
- 监督损失：`chamfer + normal + laplacian + edge + repulsion + latent`

### 本仓库新增的两个增强（基于你提供数据）

- `mask` 感知训练（对 1828 对监督样本）
  - 背景抑制（非舌体置黑）
  - 舌体 ROI 裁剪（带 margin）
  - 目的：降低嘴唇区域干扰

- `in_the_wild` 一致性训练（对 5000+ 无3D样本）
  - 约束“彩色图”和“分割图”预测的 latent 一致
  - 不需要 3D 真值
  - 目的：增强真实世界泛化

## 7. 配置重点（`configs/image2shape_4090_dense.yaml`）

关键配置项：
- `dataset.use_mask: true`
- `dataset.mask_crop: true`
- `dataset.mask_background_zero: true`
- `in_the_wild.enabled: true`
- `in_the_wild.manifest_csv: TongueDB/in_the_wild_pairs.csv`
- `loss.in_the_wild_consistency: 0.2`

你可以据此调优“嘴唇聚点”问题：
- 提高 `loss.in_the_wild_consistency`
- 增大 `dataset.mask_margin_ratio`
- 适度提升 `loss.repulsion` 与 `loss.laplacian`

## 8. 训练后评估与推理

虽然 `train.sh` 只负责训练，完整功能仍在 `run.sh`：

```bash
# 评估
./run.sh eval runs/img2shape_4090_dense/best.pt

# 单图推理
./run.sh infer runs/img2shape_4090_dense/best.pt TongueDB/images/03903.000052.png

# 可视化
./run.sh visualize TongueDB/meshes/03903.000052.obj runs/predictions/03903.000052_pred.ply
./run.sh render runs/predictions/03903.000052_pred.ply
```

## 9. 输出目录

每次训练会在 `runs/<experiment>/` 下创建时间戳目录，包含：
- `config.json`
- `normalization.json`
- `splits.csv`
- `metrics.csv`
- `tensorboard/`
- `best.pt` / `last.pt`

## 10. 当前能力边界

- 本仓库可稳定完成 2D->3D 训练与推理，但视觉结果仍受数据域差异影响。
- CloudCompare 后处理可用于工程展示，但应与模型原始输出分开汇报。
- 若后续需要完整论文链路，可继续扩展 TongueGAN 分支。 
