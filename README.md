# Tongue 2D->3D 重建实践（PyTorch）

本项目是基于 `2106.12302v1` 的工程化复现版本，针对公开 `TongueDB` 数据集实现了可训练、可评估、可视化、可对比的 2D 舌象到 3D 点云重建流程。

实现重点：
- 两阶段训练：`Point AE` -> `Image2Shape`
- 配置管理：`pydantic + yaml`
- 训练进度：`tqdm`
- 实验管理：`任务名_时间戳` 运行目录、`TensorBoard`、`metrics.csv`
- 不使用 `argparse`

## 1. 环境准备

```bash
pip install -r requirements.txt
```

推荐硬件（你当前环境可用）：
- GPU: RTX 4090 24GB
- RAM: 64GB

## 2. 数据目录

默认目录：

```text
TongueDB/
  images/*.png
  meshes/*.obj
```

通过文件名 stem 自动配对，如：
- `TongueDB/images/03903.000052.png`
- `TongueDB/meshes/03903.000052.obj`

## 3. 核心脚本

- `tongue3d/scripts/train_autoencoder.py`：第一阶段点云自编码器
- `tongue3d/scripts/train_image2shape.py`：第二阶段图像到形状
- `tongue3d/scripts/evaluate.py`：评估（CD_L1/CD_L2/F1/Precision/Recall/NormalLoss）
- `tongue3d/scripts/infer_single.py`：单例推理（支持 `sample_id + splits.csv`）
- `tongue3d/scripts/visualize_compare.py`：GT OBJ 与预测 PLY 对比图
- `tongue3d/scripts/render_blue_splat.py`：论文风格蓝色点状（高斯泼溅感）渲染

## 4. 一键流程（推荐）

```bash
./run.sh
```

可选环境变量：

```bash
AE_CONFIG=configs/autoencoder_4090_dense.yaml \
IMG_CONFIG=configs/image2shape_4090_dense.yaml \
SAMPLE_ID=03903.000052 \
./run.sh
```

`run.sh` 会自动执行：
1. 训练 AE
2. 自动注入 AE 最优权重到二阶段配置并训练 Image2Shape
3. 在 val 集评估并保存 `eval_val.json`
4. 按 `sample_id` 推理
5. 生成 GT/Pred 对比图
6. 生成蓝色点状渲染图

## 5. 分阶段手动训练

### 5.1 阶段一：AE

```bash
python -m tongue3d.scripts.train_autoencoder configs/autoencoder_4090_dense.yaml
```

### 5.2 阶段二：Image2Shape

```bash
python -m tongue3d.scripts.train_image2shape configs/image2shape_4090_dense.yaml
```

注意：二阶段会自动尝试读取 AE 根目录的 `best.pt`，若不存在会回退 `latest_run.txt` 指向的时间戳子目录。

## 6. 实验输出结构

每次运行会生成：`输出根目录/任务名_时间戳/`

典型内容：
- `config.json`
- `normalization.json`
- `splits.csv`
- `metrics.csv`
- `tensorboard/`
- `best.pt` / `last.pt`
- `visuals/epoch_050/*.png|*.ply`

同时根目录保留当前 run 的快捷入口：
- `runs/xxx/best.pt`
- `runs/xxx/last.pt`
- `runs/xxx/latest_run.txt`

## 7. TensorBoard 与日志

启动：

```bash
tensorboard --logdir runs
```

每轮记录：
- `total/chamfer/cd_l1/normal/laplacian/edge/repulsion`
- `precision/recall/fscore`
- `latent`（二阶段）
- `lr`、耗时

CSV 同步输出到每个 run 目录下的 `metrics.csv`。

## 8. 数据划分 CSV 与单例调用

训练时自动保存 `splits.csv`，格式：
- `split,sample_id,image_path,mesh_path`

单例推理（按 sample_id）：

```bash
python -m tongue3d.scripts.infer_single \
  runs/img2shape_4090_dense/best.pt \
  03903.000052 \
  runs/predictions/03903.000052_pred.ply \
  runs/img2shape_4090_dense/<某次run>/splits.csv
```

也可直接传图片路径。

## 9. 评估命令

```bash
python -m tongue3d.scripts.evaluate \
  runs/img2shape_4090_dense/best.pt \
  val \
  runs/img2shape_4090_dense/<某次run>/splits.csv \
  runs/img2shape_4090_dense/<某次run>/eval_val.json
```

输出指标：
- `CD_L2`
- `CD_L1`
- `F1@threshold`
- `Precision / Recall`
- `NormalLoss`

## 10. 可视化与论文风格蓝点渲染

### 10.1 GT/Pred 对比图

```bash
python -m tongue3d.scripts.visualize_compare \
  TongueDB/meshes/03903.000052.obj \
  runs/predictions/03903.000052_pred.ply \
  runs/compare/03903_compare.png 8192
```

### 10.2 蓝色点状（高斯泼溅感）

```bash
python -m tongue3d.scripts.render_blue_splat \
  runs/predictions/03903.000052_pred.ply \
  runs/renders/03903_pred_blue_splat.png 12000 1.0
```

## 11. 4090 参数建议（针对稀疏与平滑问题）

你当前 24GB 显存可用的稳健组合：
- `num_points: 8192`
- `batch_size: 8`
- `grad_accum_steps: 2`
- `chamfer_chunk_size: 2048`
- `freeze_decoder: false`
- `decoder_lr_scale: 0.2`
- `AE`: `laplacian=0.07 edge=0.03 repulsion=0.02`
- `Image2Shape`: `normal=0.08 laplacian=0.08 edge=0.03 repulsion=0.02 latent=1.2`

经验建议：
- 点数增加能显著改善“视觉稀疏感”，但不是唯一因素。
- 平滑性更多受 `laplacian/edge/repulsion` 和 decoder 训练策略影响。
- 二阶段如果过早收敛，优先看 `fscore` 与 `cd_l1` 是否停滞，再决定延长 epoch 或微调 loss 权重。

## 12. 与论文对齐说明

已对齐：
- 两阶段 latent 驱动框架
- 几何一致性损失（Chamfer + Normal + 正则）
- 训练过程可视化与单例定点测试

由于公开资源限制，采用神经点云解码器替代论文私有参数化资产；在工程上通过：
- 更密集采样（8192）
- repulsion 与平滑正则
- 可复现实验记录（split/metrics/tensorboard）
来尽量逼近论文展示效果。
