# Face Hallucination via Split-Attention in Split-Attention Network

This repository provides the official PyTorch implementation of the following paper:

### Paper link: [SISN-MM'21](https://dl.acm.org/doi/abs/10.1145/3474085.3475682)
### Requirement
* Python 3.7
* PyTorch >= 1.4.0 (1.5.0 is ok)
* numpy
* skimage
* imageio
* matplotlib
* tqdm
### Dataset
Please download FFHQ dataset from [here](https://github.com/NVlabs/ffhq-dataset) and CelebA dataset from [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
After download all datasets, the folder ```dataset``` should be like this (take FFHQ as an example):
```
    dataset    
    └── FFHQ
        ├── 1024X1024
            ├── HR
            └── LR
                ├── X2
                ├── X4
                └── X8
        └── 256X256
            ├── HR
            └── LR
                ├── X2
                ├── X4
                └── X8
```

### Training Model
First, you need to set the necessary parameters in the option.py such as scale, dataset_root, train_val_range, etc.
Training the model on the X4 scale as below:
```
python train.py --model SISN --scale 4
```

By default, the trained model will be saved in `./pt` directory.

### Testing model
```
python test.py --model SISN --scale 4 --pretrain <path_of_pretrained_model> --dataset_root <path_of_input_image> --save_root <path_of_result>
```

### Citation
If you find the code helpful in your resarch or work, please cite the following paper.
```
@inproceedings{lu2021face,
  title={Face Hallucination via Split-Attention in Split-Attention Network},
  author={Lu, Tao and Wang, Yuanzhi and Zhang, Yanduo and Wang, Yu and Wei, Liu and Wang, Zhongyuan and Jiang, Junjun},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={5501--5509},
  year={2021}
}
```
