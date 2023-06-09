# Docker-PyTorch2.0

This repository contains information on installing [docker](https://www.docker.com/) on a virtual/local machine, make a docker container with CUDA 11.6/11.7 and install newly released [PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/) and [PyTorch Lightning](https://www.pytorchlightning.ai/index.html).

## Step-by-step tutorial on running docker
* Install docker on your machine:
```python
pip install docker
```
* Go to [Docker Hub website](https://hub.docker.com/) and search for the necessary image, ex. cuda_11.6
* Pull it using the following command:
```python
docker pull <image name>
```
It takes some time until the pull process is completed.\

![Capture](https://user-images.githubusercontent.com/50166164/209746959-7b68f8e0-e009-442c-96bb-2c084b74692d.PNG)

* Double check whether the image is successfully installed using: 
```python
docker images
```
![Capture1](https://user-images.githubusercontent.com/50166164/209746985-5fd47b6e-c91d-4c1b-9cdd-97307150ec38.PNG)

* Run container with the installed image:
```python
docker run --gpus all -itd -p <port> --name <docker_name> <image_name>
```
where, --gpus all - if you want to access gpus;\
-p - is the port to run the docker, ex. 9110:9110, 9111:9111;\
--name - is the name for docker, pytorch2.0;\
<image_name> is the name of the image, qpod0dev/cuda_11.6

* Run the docker:

```python
docker exec -it <docker_name> /bin/bash
```
where, <docker_name> is the name of the docker, pytorch2.0

* Install libraries:
```python
pip install numpy --pre torch torchvision torchaudio --force-reinstall --extra-index-url https://download.pytorch.org/whl/nightly/cu116
```
* Double check the proper installation of the libraries and gpu availability:
```python
import torch
torch.__version__
torch.cuda.is_available()
```
It must return 2.0.0.dev20221227+cu116 and True

![Capture1](https://user-images.githubusercontent.com/50166164/209747727-88acd0c2-57fd-48e2-a89d-e1fc8720b8c1.PNG)

* Install jupyter and jupyterlab:
```python
pip install jupyter jupyterlab
```

* Run the jupyterlab:
```python
jupyter lab --ip 0.0.0.0 --allow-root --port 9110 --no-browser
```
change the port of your current running server to the port you defined before, ex: 9110. Enter the token that is generated when you run the jupyterlab

## PyTorch 2.0 Training 

* Install required libraries:
```python
pip install requirements.txt
```
* Train a model using [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database):
```python
python main.py --ds_name="mnist"
```
* Train a model using [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html):
```python
python main.py --ds_name="cifar10"
```
* Train a model using [CIFAR100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html):
```python
python main.py --ds_name="cifar100"
```
* Parameters to change: 

![Capture](https://user-images.githubusercontent.com/50166164/209887801-0e6af75b-1a7b-4e3c-8b64-c37d02251a28.PNG)

## PyTorch Lightning Training

* Train a model using custom dataset:
```python
python pl_main.py
```

* Parameters to change:

![image](https://github.com/bekhzod-olimov/Docker-PyTorch2.0/assets/50166164/271b97f1-f0cd-430d-b6b3-a2863df3f71c)

## Speedup in PyTorch 2.0
Although it is mentioned that a model can benefit up to [38% speedup](https://pytorch.org/get-started/pytorch-2.0/) (for [timm](https://github.com/rwightman/pytorch-image-models) models) when trained using torch.compile() in PyTorch 2.0, in reality model trained in PyTorch 1.13 are significantly faster in both training and validation.
Untar directories with train, validation statistics:

```python
tar -xvf stats/stats_1.13.tar
tar -xvf stats/stats_2.0.tar
```

* [GHIM-10K](https://www.kaggle.com/datasets/guohey/ghim10k)



* MNIST
![mnist_train](https://user-images.githubusercontent.com/50166164/210288075-da296ba3-0149-4cb3-a507-7aa24c29a9ac.png)
![mnist_valid](https://user-images.githubusercontent.com/50166164/210288082-6857fab1-bafb-4d2b-9a9e-87972dcee1c8.png)

* CIFAR10
![cifar10_train](https://user-images.githubusercontent.com/50166164/210288126-f04138f6-aeff-4530-92c8-5024ad375e12.png)
![cifar10_valid](https://user-images.githubusercontent.com/50166164/210288127-cd7f17ed-9eb7-4474-8276-cb49c510e42b.png)

* CIFAR100
![cifar100_train](https://user-images.githubusercontent.com/50166164/210288164-a187d973-9e4e-4fac-ae29-7e5d0ffafe3d.png)
![cifar100_valid](https://user-images.githubusercontent.com/50166164/210288165-658b1a6f-d228-45c3-baa4-641461a0eded.png)

## Accuracy in PyTorch 2.0
Accuracy difference between PyTorch 2.0 and PyTorch 1.13 is not very noticeable. The experiments on the three datasets did not show great changes in validation accuracy.

* MNIST
![mnist_acc](https://user-images.githubusercontent.com/50166164/210288202-f7f233a9-f1b3-4997-b545-da627a38d407.png)

* CIFAR10
![cifar10_acc](https://user-images.githubusercontent.com/50166164/210288209-1787e4dc-d18a-4a3c-8105-603899664fab.png)

* CIFAR100
![cifar100_acc](https://user-images.githubusercontent.com/50166164/210288225-5210163c-065c-4656-b7c3-b97ebe07c09a.png)













