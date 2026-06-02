# Installation
Please follow the instructions to set up the envrionment.

## 1. Create conda environment
```bash
conda create -n gf3d python=3.8.16
conda activate gf3d
```

## 2. (Option 1) Install PyTorch (CUDA 11.8) + Compatible MMLab packages
```bash
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install openmim
mim install mmcv==2.0.1
mim install mmdet==3.0.0
mim install mmsegmentation==1.0.0
mim install mmdet3d==1.1.1
```

## 2. (Option 2) Install PyTorch (CUDA 12.1) + Compatible MMLab packages
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install openmim
mim install mmcv==2.1.0
mim install mmdet==3.2.0
mim install mmsegmentation==1.2.1
mim install mmdet3d==1.3.0
```

## 3. Install other packages
```bash
pip install spconv-cu117
pip install timm
```

## 4. Install custom CUDA ops for 2d deformable attention and Gaussian-to-voxel splatting developed by [GaussianFormer](https://github.com/huang-yh/GaussianFormer)
```bash
cd model/encoder/gaussian_encoder/ops && pip install -e .
cd model/head/localagg && pip install -e .
```
Note: The default number of class in localagg is set as 18 which corresponds to the nuScenes-SurroundOcc dataset. If you wish to run experiments on RELLIS3D-WildOcc dataset, please modify NUM_CHANNELS to be 9 in the config.h file, and recompile it with ```cd model/head/localagg && pip install -e .```.


## 5. Install custom CUDA ops for 3D deformable attention developed by [DFA3D](https://github.com/IDEA-Research/3D-deformable-attention)
```bash
git clone https://github.com/IDEA-Research/3D-deformable-attention.git
cd 3D-deformable-attention/
cd DFA3D
bash setup.sh 0
# Check Installation
cd ../
python unittest_DFA3D.py
```
Note on setup.py: You may need to modify line 161 to be: extra_compile_args['cxx'] = ['-std=c++17'] and line 185 to be: extra_compile_args['nvcc'] += ['-std=c++17'].
