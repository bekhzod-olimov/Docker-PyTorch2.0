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
It takes some time until the pull process is completed.
![Capture](https://user-images.githubusercontent.com/50166164/209746959-7b68f8e0-e009-442c-96bb-2c084b74692d.PNG)\

* Double check whether the image is successfully installed using: 
```python
docker images
```
![Capture1](https://user-images.githubusercontent.com/50166164/209746985-5fd47b6e-c91d-4c1b-9cdd-97307150ec38.PNG)\

* Run container with the installed image:
```python
docker run --gpus all -itd -p <port> -p <port> --name <docker_name> <image_name>
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


