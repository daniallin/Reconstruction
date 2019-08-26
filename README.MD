using multi-task neural network to complete 3D reconstruction.

## 1. Multi-task of LIN
The multi-task means 3 tasks: depth estimation using single image, semantic segmentation and vision odometry.

### Running process
1. Download the virtual-kitti dataset, then run the `utils/vkitti_preprocess.py` to get the structural data. The pickle files will be saved in `../dataset/vkitti`.
   > Remember to modify the `path` to your own path

1.1. `create_npy.py` : https://github.com/VisualComputingInstitute/vkitti3D-dataset
   
2. Run the `train_mtn.py`.


### Request
PyTorch 1.2+



## 2. Second Part
Learning the multi-task neural network. Using the source code from MTAN.

### 2.1 Files introduction
- models/mtan.py: the implementation of MTAN, almost the same as `https://github.com/lorenmt/mtan`
- train_mtan.py: the training script of MTAN

### 2.2 Reference
- [End-to-End Multi-Task Learning with Attention](https://github.com/lorenmt/mtan)

