python -m tongue3d.scripts.train_autoencoder configs/autoencoder_4090_dense.yaml
python -m tongue3d.scripts.train_image2shape configs/image2shape_4090_dense.yaml
python -m tongue3d.scripts.infer_single runs/img2shape_4090_dense/best.pt TongueDB/images/03903.000052.png runs/predictions/03903.000052_dense.ply
