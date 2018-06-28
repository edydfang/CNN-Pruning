# PyTorch implementation of CNN Pruning

This repo is a modified version of https://github.com/jacobgil/pytorch-pruning.

This is an implementation of the paper [\[1611.06440 Pruning Convolutional Neural Networks for Resource Efficient Inference\]](https://arxiv.org/abs/1611.06440).

This demonstrates pruning a VGG16 based classifier that classifies a small dog/cat dataset.


This was able to reduce the CPU runtime by x3 and the model size by x4.

For more details you can read the [blog post](https://jacobgil.github.io/deeplearning/pruning-deep-learning).

At each pruning step 512 filters are removed from the network.


## Usage

Tested Environment:

- CUDA 9.1
- Python 3.6 with PyTorch

This repository uses the PyTorch ImageFolder loader, so it assumes that the images are in a different directory for each category.

```
├── dataset.py
├── finetune.py
├── prune.py
├── test
│   ├── cats
│   └── dogs
└── train
    ├── cats
    └── dogs
```

The images were taken from [here](https://www.kaggle.com/c/dogs-vs-cats) but you should try training this on your own data and see if it works!

Training:
`python finetune.py --train`

Pruning:
`python finetune.py --prune`

## TBD


 - Change the pruning to be done in one pass. Currently each of the 512 filters are pruned sequentually. 
	`
	for layer_index, filter_index in prune_targets:
			model = prune_vgg16_conv_layer(model, layer_index, filter_index)
		`


 	This is inefficient since allocating new layers, especially fully connected layers with lots of parameters, is slow.
	
	In principle this can be done in a single pass.



 - Change prune_vgg16_conv_layer to support additional architectures.
 	The most immediate one would be VGG with batch norm.

## Reference
https://github.com/jacobgil/pytorch-pruning


