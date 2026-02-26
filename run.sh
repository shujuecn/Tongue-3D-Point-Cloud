python -m tongue3d.scripts.train_autoencoder configs/autoencoder_4090_dense.yaml
python -m tongue3d.scripts.train_image2shape configs/image2shape_4090_dense.yaml
python -m tongue3d.scripts.visualize_compare TongueDB/meshes/03903.000052.obj runs/predictions/03903.000052_dense.ply runs/compare/03903_compare.png
