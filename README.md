# Tongue 2D->3D Mask-Guided Baseline

本分支基于 `codex/baseline`，目标不是做 2D/3D 分割任务，而是测试：

给现有 `1829` 对 2D-3D 配对样本补充分割 mask 后，是否能作为额外先验提升 3D 点云生成与重建效果。

## 1. 这条分支具体改了什么

- 保留 `baseline` 的两阶段训练结构：
  - 阶段一：点云 AE
  - 阶段二：Image2Shape
- 不引入额外分割头，不引入多任务损失，不改 3D 监督目标。
- 新增 `dataset.mask_as_channel`：
  - 训练/评估/推理时，将 `RGB` 与二值 `mask` 拼成 `4` 通道输入。
  - 也就是让 image encoder 直接看到“舌体位置先验”。
- ResNet-50 首层卷积已改为支持 `4` 通道输入，额外通道用预训练权重均值初始化。

这意味着本分支测的是：

`mask 作为输入先验` 是否有助于 3D 重建。

不是：

- 预测 2D mask
- 预测 3D 分割
- 多任务学习

## 2. 实验对照设计

本分支保留两套二阶段配置：

### 2.1 纯监督对照

文件：

`configs/image2shape_4090_dense.yaml`

特点：

- `dataset.use_mask: false`
- `dataset.mask_as_channel: false`
- `in_the_wild.enabled: false`

这是不使用 mask 的纯监督基线。

### 2.2 Mask-Guided 基线

文件：

`configs/image2shape_4090_mask_guided.yaml`

特点：

- `dataset.use_mask: false`
- `dataset.mask_as_channel: true`
- `dataset.mask_crop: false`
- `in_the_wild.enabled: false`

这里不裁剪、不置零背景，只把 mask 当作第 4 个输入通道。

## 3. 数据要求

纯监督对照只需要：

```text
TongueDB/
  images/*.png
  meshes/*.obj
```

Mask-Guided 额外需要：

```text
TongueDB/
  masks/*.png
```

要求：

- `images / meshes / masks` 同名同 stem
- mask 为单通道或可转灰度的二值图
- 默认阈值 `127`

例如：

```text
TongueDB/images/03903.000052.png
TongueDB/meshes/03903.000052.obj
TongueDB/masks/03903.000052.png
```

## 4. 推荐训练流程

### 4.1 阶段一：先训练 AE

这一步与是否使用 mask 无关，只训练 3D 点云自编码器：

```bash
./train.sh ae configs/autoencoder_4090_dense.yaml
```

### 4.2 阶段二 A：跑纯监督对照

```bash
./train.sh img configs/image2shape_4090_dense.yaml runs/ae_4090_dense/best.pt
```

### 4.3 阶段二 B：跑 Mask-Guided 基线

```bash
./train.sh img configs/image2shape_4090_mask_guided.yaml runs/ae_4090_dense/best.pt
```

如果 WSL 下需要提速：

```bash
TONGUE3D_ALLOW_WSL_WORKERS=1 TONGUE3D_NUM_WORKERS=4 \
./train.sh img configs/image2shape_4090_mask_guided.yaml runs/ae_4090_dense/best.pt
```

## 5. 评估与推理

### 5.1 评估纯监督对照

```bash
./run.sh eval runs/img2shape_4090_dense/best.pt
```

### 5.2 评估 Mask-Guided 基线

```bash
./run.sh eval runs/img2shape_4090_mask_guided/best.pt
```

### 5.3 单例推理

纯监督模型：

```bash
./run.sh infer runs/img2shape_4090_dense/best.pt TongueDB/images/03903.000052.png
```

Mask-Guided 模型：

```bash
./run.sh infer runs/img2shape_4090_mask_guided/best.pt TongueDB/images/03903.000052.png
```

注意：

- 纯监督模型只需要图像
- Mask-Guided 模型会自动查找同名 mask：
  - `TongueDB/masks/03903.000052.png`
- 如果 mask 不存在，会直接报错，因为这个实验分支假定 mask 是输入先验的一部分

## 6. 实践指导

### 6.1 应该怎么比较

建议固定以下条件不变：

- 使用同一个 AE checkpoint
- 使用同一份 `splits.csv`
- 使用同一套评估命令
- 只改变二阶段配置

建议对比：

1. `runs/img2shape_4090_dense/best.pt`
2. `runs/img2shape_4090_mask_guided/best.pt`

重点看：

- `CD_L2`
- `CD_L1`
- `F1@0.01`
- CloudCompare 中的整体形状是否更接近 GT
- 嘴唇附近是否仍然出现明显点云堆积

### 6.2 这个 baseline 的预期收益

如果 mask 有帮助，通常会表现为：

- 舌体区域定位更稳定
- 背景和口唇干扰减弱
- 点云更少向口周漂移

### 6.3 如果效果没有提升，常见原因

常见失败原因是这几个：

1. mask 只提供了轮廓位置，但没有补充深度/厚度信息  
2. mask 质量不稳定，反而给 encoder 引入噪声  
3. 口唇、牙齿、阴影等上下文本身对 3D 形状是有帮助的，硬加 mask 可能让模型过度依赖前景轮廓  
4. 4 通道输入虽然更聚焦，但不一定比原始 RGB 更利于恢复细节

所以这条分支的意义是：

先用一个最干净的方式验证“mask 先验”是否值得继续深入。

如果这一步都没有提升，就没必要马上继续加更复杂的 mask 相关机制。

## 7. 代码结构对齐关系

这条分支依然和主工程保持同一套入口：

- `run.sh`
- `train.sh`
- `tongue3d/scripts/train_autoencoder.py`
- `tongue3d/scripts/train_image2shape.py`
- `tongue3d/scripts/evaluate.py`
- `tongue3d/scripts/infer_single.py`

所以后续切回 `main` 或切到别的实验分支时，使用方式不需要重学。

## 8. 切换建议

切回纯监督基线分支：

```bash
git checkout codex/baseline
```

切回主干：

```bash
git checkout main
```

切回当前 mask-guided 实验分支：

```bash
git checkout codex/dev-mask-guidance
```

## 9. 结论定位

这条分支不是最终方案，而是一个干净的实验基线：

它只回答一个问题：

在不引入 5000+ 额外图像、不引入 TongueGAN、不引入分割多任务的前提下，给配对数据增加 mask 输入先验，能不能提升 3D 重建效果。
