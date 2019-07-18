# Pixel2Mesh

## TODOS

- [x] Reorganize everything
- [x] Add batch support
- [x] Add mesh visualization in tensorboard
- [x] Add normal loss
- [x] Add evaluation
- [ ] Train to see if everything works (not sure why it doesn't)

## Milestones

Three = Plane + Lamp + Chair (Following Tong-ZHAO)

Source | Dataset | Chamfer | F1 (1x) | F1 (2x) 
-------|---------|---------|---------|--------
Paper  | Full    | 0.591   | 59.72   | 74.19  
Paper  | Plane   | 0.477   | 71.12   | 81.38
Tong-ZHAO-Pytorch (partial) | Plane | 0.455 | 63.08 | 76.58
Official Model (evaluated in MeshRCNN) | Full | 0.591 | 59.72 | 74.19
Pixel2Mesh+ (MeshRCNN) | Full | 0.284 | 75.83 | 86.63
baseline_lr_1e-4_zthresh_resnet | Three | 0.533 | 57.67 | 73.83
baseline_lr_1e-4 | Three | 0.558 | 54.71 | 72.26

More models still training...

## Environment Preparation

It seems that installing the following packages will automatically resolve everything in need.

```
pytorch
torchvision
cudatoolkit=9.0
opencv
scipy
scikit-image
easydict
pyyaml
tensorboardx
tensorboard
tensorflow
trimesh
shapely
```

Do `git submodule update --init` to get NeuralRenderer ready.

In `external/chamfer` and `external/neural_renderer` do:

```
python setup.py install
```

## Dataset Preparation

Everything you need to find and orginize are listed as below:

```
datasets/data
├── ellipsoid
│   ├── face1.obj
│   ├── face2.obj
│   ├── face3.obj
│   └── info_ellipsoid.dat
├── pretrained
│   ├── resnet50-19c8e357.pth
│   └── vgg16-397923af.pth
└── shapenet
    ├── data
    │   ├── 02691156
    │   │   └── 3a123ae34379ea6871a70be9f12ce8b0_02.dat
    │   └── 02828884
    └── meta
        ├── shapenet.json
        ├── shapenet_labels.pkl
        ├── test_plane.txt
        ├── test_small.txt
        ├── train_plane.txt
        └── train_small.txt
```

## Training

Train locally:

```
python entrypoint_train.py --name whatever_you_want_to_call_it --options experiments/baseline_single.yml
```

Train with slurm:

```
python entrypoint_train.py --name whatever_you_want_to_call_it --options experiments/baseline.yml
```
