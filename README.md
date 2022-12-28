# Docker-PyTorch2.0

This repository contains information on installing [docker](https://www.docker.com/) on a virtual/local machine, make a docker container with CUDA 11.6/11.7 and install newly released [PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/).

### Step-by-step tutorial on running docker
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
* 
