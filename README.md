# Experiments on model collapse

This repository is based on <https://github.com/mcclow12/ncsn-pytorch>, and contains the following experiments: 

- Training on Fashion-MNIST.
- Tuning on $\sigma$ values to slow down memorization.
- Training on _large U-Net_ with increased embedding size and dropout layers.

## Installation
This project was made with **Python 3.13**, to install dependencies you can use uv (<https://docs.astral.sh/uv/>):

```shell
uv init -p 3.13
uv venv 
uv add -r requirements_with_versions.txt
```

If the version resolution fails or if you prefer using another python version with pip, you can use the `requirements.txt`:

```shell
pip install -r requirements.txt
```


## Training models
To train models on Fashion-MNIST you can run:

```shell
python scripts/train_fashion_mnist.py --num_iters 40000 --save_every 5000
```

The parser let you choose the following hyperparameters: `lr, beta1, beta2, batch_size, num_iters, ngf, logit_transform, random_flip, dropout, weight_decay, ema_rate` and less important parameters related to the task: `image_size, channels, num_classes, checkpoints_folder, samples_folder`. 

## Testing models
Once models have been trained, you can sample with: 

```shell
python scripts/sample_unet.py --checkpoint_name mnist_1.pt --samples_name mnist_sample.png --ngf 64
```

To use the _original U-Net_, you can keep the default `--ngf 64`, but to sample the _large U-Net_ you need to use `--ngf 96`.