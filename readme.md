# Pixel2Mesh

## TODOS

- [x] Reorganize everything
- [x] Add batch support
- [x] Add mesh visualization in tensorboard
- [ ] Add normal loss
- [ ] Add evaluation
- [ ] Train to see if everything works

## Environment Preparation

Basically, you will need:

* Python 3.5+
* PyTorch 1.1+
* Cuda 9.0+
* OpenCV
* Scikit Image
* EasyDict
* PyYAML
* TensorboardX

Note that this is not guaranteed to be a complete list of dependencies.

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
