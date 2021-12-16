# EfficientLPS: Efficient LiDAR Panoptic Segmentation
EfficientLPS is a state-of-the-art top-down approach for LiDAR panoptic segmentation, where the goal is to assign semantic labels (e.g., car, road, tree and so on) to every point in the input LiDAR point cloud as well as instance labels (e.g. an id of 1, 2, 3, etc) to points belonging to thing classes.

![Illustration of EfficientLPS](/images/intro.png)

This repository contains the **PyTorch implementation** of our T-RO'2021 paper [EfficientLPS: Efficient LiDAR Panoptic Segmentation](https://arxiv.org/abs/2102.08009). The repository builds on [mmdetection](https://github.com/open-mmlab/mmdetection) and [gen-efficientnet-pytorch](https://github.com/rwightman/gen-efficientnet-pytorch) codebases.

If you find the code useful for your research, please consider citing our paper:
```
@article{sirohi2021efficientlps,
  title={EfficientLPS: Efficient LiDAR Panoptic Segmentation},
  author={Sirohi, Kshitij and Mohan, Rohit and B{\"u}scher, Daniel and Burgard, Wolfram and Valada, Abhinav},
  journal={arXiv preprint arXiv:2102.08009},
  year={2021}
}
```
## Demo
https://rl.uni-freiburg.de/research/lidar-panoptic

## System Requirements
* Linux 
* Python 3.7
* PyTorch 1.7
* CUDA 10.2
* GCC 7 or 8

**IMPORTANT NOTE**: These requirements are not necessarily mandatory. However, we have only tested the code under the above settings and cannot provide support for other setups.

## Installation
a. Create a conda virtual environment from the provided environment.yml and activate it.
```shell
git clone https://github.com/robot-learning-freiburg/EfficientLPS.git
cd EfficientLPS
conda env create -n efficientLPS_env --file=environment.yml
conda activate efficientLPS_env
```
b. Install all other dependencies using pip:
```bash
pip install -r requirements.txt
```
c. Install EfficientNet implementation
```bash
cd efficientNet
python setup.py develop
```
d. Install EfficientLPS implementation
```bash
cd ..
python setup.py develop
```

### Todo

- [ ] Add inference script  
- [ ] Add support for Panoptic nuScenes  
- [ ] Add panoptic periphery loss
- [ ] Add pre-trained models
- [ ] Add training and evaluation instructions

## Acknowledgements
We have used utility functions from other open-source projects. We espeicially thank the authors of:
- [mmdetection](https://github.com/open-mmlab/mmdetection)
- [gen-efficientnet-pytorch](https://github.com/rwightman/gen-efficientnet-pytorch)
- [semantic-kitti-api](https://github.com/PRBonn/semantic-kitti-api)

## Contacts
* [Abhinav Valada](https://rl.uni-freiburg.de/people/valada)
* [Rohit Mohan](https://github.com/mohan1914)

## License
For academic usage, the code is released under the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) license. For any commercial purpose, please contact the authors.
