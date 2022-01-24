# Compact Transformers

	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/escaping-the-big-data-paradigm-with-compact/image-classification-on-flowers-102)](https://paperswithcode.com/sota/image-classification-on-flowers-102?p=escaping-the-big-data-paradigm-with-compact)

Preprint Link: [Escaping the Big Data Paradigm with Compact Transformers
](https://arxiv.org/abs/2104.05704)

By [Ali Hassani<sup>[1]</sup><span>&#42;</span>](https://alihassanijr.com/),
[Steven Walton<sup>[1]</sup><span>&#42;</span>](https://stevenwalton.github.io/),
[Nikhil Shah<sup>[1]</sup>](https://itsshnik.github.io/),
[Abulikemu Abuduweili<sup>[1]</sup>](https://github.com/Walleclipse),
[Jiachen Li<sup>[1,2]</sup>](https://chrisjuniorli.github.io/), 
and
[Humphrey Shi<sup>[1,2,3]</sup>](https://www.humphreyshi.com/)


<small><span>&#42;</span>Ali Hassani and Steven Walton contributed equal work</small>

In association with SHI Lab @ University of Oregon<sup>[1]</sup> and
UIUC<sup>[2]</sup>, and Picsart AI Research (PAIR)<sup>[3]</sup>

 

## Other implementations & resources
**[PyTorch blog]**: check out our [official blog post with PyTorch](https://medium.com/pytorch/training-compact-transformers-from-scratch-in-30-minutes-with-pytorch-ff5c21668ed5) to learn more about our work and vision transformers in general.

**[Keras]**: check out [Compact Convolutional Transformers on keras.io](https://keras.io/examples/vision/cct/) by [Sayak Paul](https://github.com/sayakpaul).

**[vit-pytorch]**: CCT is also available through [Phil Wang](https://github.com/lucidrains)'s [vit-pytorch](https://github.com/lucidrains/vit-pytorch), simply use ```pip install vit-pytorch```


# Abstract
 
#### ViT-Lite: Lightweight ViT 
Different from [ViT](https://arxiv.org/abs/2010.11929) we show that <i>an image 
is <b>not always</b> worth 16x16 words</i> and the image patch size matters.
Transformers are not in fact ''data-hungry,'' as the authors proposed, and
smaller patching can be used to train efficiently on smaller datasets.

#### CVT: Compact Vision Transformers
Compact Vision Transformers better utilize information with Sequence Pooling post 
encoder, eliminating the need for the class token while achieving better
accuracy.

#### CCT: Compact Convolutional Transformers
Compact Convolutional Transformers not only use the sequence pooling but also
replace the patch embedding with a convolutional embedding, allowing for better
inductive bias and making positional embeddings optional. CCT achieves better
accuracy than ViT-Lite and CVT and increases the flexibility of the input
parameters.

 

# How to run

## Install locally

### Requirements

Python 3.7

Our base model is in pure PyTorch and Torchvision. No extra packages are required.
Please refer to [PyTorch's Getting Started](https://pytorch.org/get-started/locally/) page for detailed instructions.

Here are some of the models that can be imported from `src` (full list available in [Variants.md](Variants.md)):

 

You can simply import the names provided in the **Name** column:
```python3
from src import cct_14_7x2_384
model = cct_14_7x2_384(pretrained=True, progress=True)
```
The config files are provided both to specify the training settings and hyperparameters, 
and allow easier reproduction.

Please note that the models missing pretrained weights will be updated soon. They were previously 
trained using our old training script, and we're working on training them again with the new script 
for consistency.


python train.py -c configs/datasets/cifar100.yml --model manifold_cvt_6_4_32  --gpu 1 --log-wandb  ./data/cifar100
You could even create your own models with different image resolutions, positional embeddings, and number of classes:
```python3
from src import cct_14_7x2_384, cct_7_7x2_224_sine
model = cct_14_7x2_384(img_size=256)
model = cct_7_7x2_224_sine(img_size=256, positional_embedding='sine')
```
Changing resolution and setting `pretrained=True` will interpolate the PE vector to support the new size, 
just like ViT.

These models are also based on experiments in the paper. You can create your own versions:
```python3
from src import cct_14
model = cct_14(arch='custom', pretrained=False, progress=False, kernel_size=5, n_conv_layers=3)
```

You can even go further and create your own custom variant by importing the class CCT.

All of these apply to CVT and ViT as well.


## Training

[timm](https://github.com/rwightman/pytorch-image-models) is recommended for image classification training 
and required for the training script provided in this repository:
### Distributed training
```shell
./dist_classification.sh $NUM_GPUS -c $CONFIG_FILE /path/to/dataset
```

You can use our training configurations provided in `configs/`:
```shell
./dist_classification.sh 2 -c configs/imagenet.yml --model manifold_cct_7_7x2_224 /home/papastrat/Desktop/imagenet
```

### Non-distributed training
```shell
python train.py -c configs/datasets/cifar10.yml --model cct_7_3x1_32 /path/to/cifar10
```
```
python train.py -c configs/datasets/cifar10.yml --model grassmanian_vit_6_4_32 --gpu 0 --log-wandb ./data/CIFAR-10-images-master/
```

### Models and config files
We've updated this repository and moved the previous training script and the checkpoints associated 
with it to `examples/`. The new training script here is just the `timm` training script. We've provided
the checkpoints associated with it in the next section, and the hyperparameters are all provided in
`configs/pretrained` for models trained from scratch, and `configs/finetuned` for fine-tuned models.

# Results
Type can be read in the format `L/PxC` where `L` is the number of transformer
layers, `P` is the patch/convolution size, and `C` (CCT only) is the number of
convolutional layers.

## CIFAR-10 and CIFAR-100

 Download Cifar-10 and CIFAR-100 datasets using 

```python
$ pip install cifar2png
```

### CIFAR-10

`$ cifar2png cifar10 path/to/cifar10png`


### CIFAR-10 with naming option

`$ cifar2png cifar10 path/to/cifar10png --name-with-batch-index`


### CIFAR-100

`$ cifar2png cifar100 path/to/cifar100png`


### CIFAR-100 with superclass

`$ cifar2png cifar100superclass path/to/cifar100png`


## Structure of output directory

### CIFAR-10 and CIFAR-100

PNG images of CIFAR-10 are saved in 10 subdirectories of each label under the `test` and `train` directories as below.  
(CIFAR-100 are saved in the same way with 100 subdirectories)

```bash
$ tree -d path/to/cifar10png
path/to/cifar10png
├── test
│   ├── airplane
│   ├── automobile
│   ├── bird
│   ├── cat
│   ├── deer
│   ├── dog
│   ├── frog
│   ├── horse
│   ├── ship
│   └── truck
└── train
    ├── airplane
    ├── automobile
    ├── bird
    ├── cat
    ├── deer
    ├── dog
    ├── frog
    ├── horse
    ├── ship
    └── truck
```

```bash
$ tree path/to/cifar10png/test/airplane
path/to/cifar10png/test/airplane
├── 0001.png
├── 0002.png
├── 0003.png
(..snip..)
├── 0998.png
├── 0999.png
└── 1000.png
```




## Flowers-102

 

## ImageNet

 


## Acknowledgments

Thanks to   [SHI Lab](https://github.com/SHI-Labs) for the awesome code for the transformers,
which I have used for my research implementations.
# Citation

Code obtained from:
```bibtex
@article{hassani2021escaping,
	title        = {Escaping the Big Data Paradigm with Compact Transformers},
	author       = {Ali Hassani and Steven Walton and Nikhil Shah and Abulikemu Abuduweili and Jiachen Li and Humphrey Shi},
	year         = 2021,
	url          = {https://arxiv.org/abs/2104.05704},
	eprint       = {2104.05704},
	archiveprefix = {arXiv},
	primaryclass = {cs.CV}
}
```
