# Pixel2Mesh

## TODOS

- [x] Reorganize everything
- [x] Add batch support
- [x] Add mesh visualization in tensorboard
- [x] Add normal loss
- [x] Add evaluation
- [ ] Train to see if everything works (not sure why it doesn't)

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
