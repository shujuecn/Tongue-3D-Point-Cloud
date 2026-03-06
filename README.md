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
- `tongue3d/scripts/infer_single.py`：单例推理（支持任意 2D 图像路径；也支持 `sample_id + splits.csv`）
- `tongue3d/scripts/visualize_compare.py`：GT OBJ 与预测 PLY 对比图
- `tongue3d/scripts/render_blue_splat.py`：论文风格蓝色点状（高斯泼溅感）渲染

## 4. `run.sh` 子命令（训练/评估/可视化/推理解耦）

查看全部命令：

```bash
./run.sh help
```

### 4.1 训练 AE

```bash
./run.sh train-ae
./run.sh train-ae configs/autoencoder_4090_dense.yaml
```

### 4.2 训练 Image2Shape

默认使用配置文件里的 `autoencoder_checkpoint`：

```bash
./run.sh train-img
./run.sh train-img configs/image2shape_4090_dense.yaml
```

也可临时覆盖 AE checkpoint：

```bash
./run.sh train-img configs/image2shape_4090_dense.yaml runs/ae_4090_dense/best.pt
```

### 4.3 评估

```bash
./run.sh eval runs/img2shape_4090_dense/best.pt
./run.sh eval runs/img2shape_4090_dense/best.pt val runs/img2shape_4090_dense/<某次run>/splits.csv runs/img2shape_4090_dense/<某次run>/eval_val.json
```

### 4.4 单例推理（任意 2D 图像，不需要 `splits.csv`）

```bash
./run.sh infer runs/img2shape_4090_dense/best.pt TongueDB/images/03903.000052.png
./run.sh infer runs/img2shape_4090_dense/best.pt /path/to/your_image.png runs/predictions/custom_pred.ply
```

### 4.5 可视化与论文风格蓝点渲染

```bash
./run.sh visualize TongueDB/meshes/03903.000052.obj runs/predictions/03903.000052_pred.ply runs/compare/03903_compare.png 8192
./run.sh render runs/predictions/03903.000052_pred.ply runs/renders/03903_pred_blue_splat.png 12000 1.0
```

## 5. 分阶段直接 Python 调用（等价）

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

## 8. 数据划分 CSV（仅 `sample_id` 推理时需要）

训练时自动保存 `splits.csv`，格式：
- `split,sample_id,image_path,mesh_path`

当你传入图片路径时，不需要 `splits.csv`。  
只有在“按 `sample_id` 推理”时才需要 `splits.csv`（用于把 sample_id 解析成图片路径）：

```bash
python -m tongue3d.scripts.infer_single \
  runs/img2shape_4090_dense/best.pt \
  03903.000052 \
  runs/predictions/03903.000052_pred.ply \
  runs/img2shape_4090_dense/<某次run>/splits.csv
```

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
- `runtime.num_workers: 0`（WSL 环境建议；Linux 原生可尝试 4~8）
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

## 13. 模型运作流程（面向 2D 图像背景）

### 13.1 两阶段为什么要拆开

目标是做 `2D 图像 -> 3D 点云`。直接端到端学习很难收敛，所以先拆成两步：
- 阶段一（AE）：先让模型学会“3D 舌体形状空间”。
- 阶段二（Image2Shape）：再让模型学会“从 2D 图像映射到这个 3D 形状空间”。

### 13.2 阶段一：Point AutoEncoder（只看 3D）

输入：真实舌体网格采样后的点云（带法向）。  
输出：重建点云（同样数量的点）。

它学到两件事：
- `encoder`：把点云压缩成 latent 向量。
- `decoder`：把 latent 还原回点云。

训练损失主要包括：
- `Chamfer`：几何位置接近。
- `Normal`：局部朝向一致。
- `Laplacian/Edge/Repulsion`：抑制噪声、控制局部平滑和点分布。

阶段一的核心产物：一个可用的 3D 解码器（`best.pt`）。

### 13.3 阶段二：Image2Shape（2D -> latent -> 3D）

输入：舌象图片。  
输出：点云。

内部路径：
1. 图像编码器提取图像特征。
2. `mapper` 预测 latent。
3. 复用阶段一的 decoder 把 latent 解码成 3D 点云。

这个阶段除了几何损失，还用 `latent loss` 约束预测 latent 接近 AE 的目标 latent。

### 13.4 推理时到底用了哪部分

推理使用的是第二阶段 checkpoint（`img2shape` 的 `best.pt`）。  
但这个 checkpoint 里已经包含（或依赖）阶段一训练好的 decoder，因此两阶段能力共同决定最终效果。

## 14. 后处理与“是否作弊”

### 14.1 CloudCompare 平滑/重采样/重建算不算作弊

不算作弊，但必须明确口径：
- 如果你报告的是“网络原始输出能力”，就只能用模型直接输出点云评估。
- 如果你报告的是“工程可用结果”，可以加后处理，但要单独标注为 `post-processed`，并同时给原始结果。

推荐实践：
- 报两组指标：`raw` 与 `raw + post`。
- 可视化也放两组，避免误解成模型本体提升。

### 14.2 论文风格图需要配准和重建参数，是否影响公平性

会影响“视觉观感公平性”，但不影响“原始模型能力”定义。  
建议把流程拆开汇报：
- 模型能力：在统一坐标、统一采样预算下比较 CD/F1 等指标。
- 展示能力：在固定后处理参数下输出论文风格图。

## 15. 当前复现完成度与能力瓶颈

### 15.1 完成度（工程复现视角）

按公开信息可复现的模块评估，当前大致在 `70%~80%`：
- 已完成：两阶段训练、评估、单例推理、可视化、可复现实验记录。
- 未完全对齐：论文私有数据清洗细节、相机/姿态先验、更强形状先验与后处理流水线细节。

### 15.2 直接限制能力的关键点

- 2D 图像到 3D 的先验不足：嘴唇区域纹理强、梯度显著，模型容易把点“吸”到嘴唇邻域。
- 训练目标偏全局：Chamfer 在局部区域可能出现“看起来近，但语义位置错”。
- 缺少显式舌体 ROI 约束：未强制模型聚焦舌体可见区域。
- 数据量和姿态分布有限：容易对高频区域（嘴唇边缘）过拟合。

### 15.3 优先级最高的改进（先做这 5 项）

1. 加舌体分割先验：先做 tongue mask，再裁剪/加权训练，弱化嘴唇区域影响。  
2. 区域加权损失：对 mouth boundary 附近提高惩罚，减少“点吸附到嘴唇”。  
3. 提升阶段二 latent 约束：增大 `latent loss` 权重并做 schedule，防止图像分支漂移。  
4. 提升输入与采样一致性：统一口腔 ROI 对齐策略，减少姿态与尺度抖动。  
5. 增加 hard-case 验证集：专门监控“嘴唇聚点率”，把它作为早停/选模指标之一。

### 15.4 一个实用诊断指标（建议加到评估）

可以新增 `lip concentration ratio`：
- 在标注或近似 ROI 下，统计预测点落在嘴唇区域的比例。
- 该比例过高通常与“嘴唇聚点”现象强相关。
