# Installation
Can either install all required packages onto your env or use a singularity image (directions below [Using Singularity (Build)](#using-singularity-build) and [Slurm with singularity](#slurm-with-singularity)). If installing into your env, use the following command and run using [Run Example](#run-example)
```
pip install -e .
```

# Run Example

run the h3.py config for 300 QMC steps

```
spinornet --config='./spinornet/configs/h3.py' --config.optim.iterations=300
```

# Using Singularity (Build)

We need to first build the image on a machine where `docker` and `singualrity` are both available. `root` permission is not required, but typically you need to be added to a `docker` user group by your cluster admin to be able to build a docker image (see [here](https://stackoverflow.com/questions/67261873/building-docker-image-as-non-root-user) and [here](https://cloudyuga.guru/blogs/manage-docker-as-non-root-user/)). 

The first step is to run a Docker container `registry` that sets up a local docker image registry at `localhost:5000`:

```shell
docker run -d -p 5000:5000 --restart=always --name registry registry:2
```

Then, build the docker image from `Dockerfile`. Log outputs are available at `docker_build.log`:

```shell
docker build -t localhost:5000/spinornet --progress=plain  . &> docker_build.log
```

Push the image to `localhost:5000`.

```shell
docker push localhost:5000/spinornet
```

To test your docker

```shell
docker run --name spinornet_test -dt --gpus all --mount type=bind,source=./,target=/home/spinornet localhost:5000/spinornet
```

```shell
docker exec -it spinornet_test /bin/bash
```

The next step builds the singularity image. The singularity image is pretty large (~6G) so it is recommended that you build it at a non-backed-up folder on the cluster, typically some kind of scratch folder.

```shell
SINGULARITY_NOHTTPS=true singularity build spinornet.sif docker://localhost:5000/spinornet:latest
```

We can now safely remove the `registry` container.

```shell
docker container stop registry && docker container rm registry
```

# Slurm with singularity

```shell
singularity exec --no-home --nv --bind .:/home/spinornet --pwd /home/spinornet $SCRATCH/spinornet.sif /bin/bash -c \
&& 'python run_spinornet.py --config=./spinornet/configs/h3.py --config.optim.iterations=300'
```

# Testing / vscode with singularity

For testing/analysis, can open a vscode remote with singularity. Suggest to follow the instructions here: [here](https://github.com/microsoft/vscode-remote-release/issues/3066#issuecomment-1019500216). When that is setup, you can open an ipynb, click Select Kernel. For the first time "selecting kernel", will need to "install suggested extensions Python + Jupyter". After installation, click on "select kernel" again and should be able to select the singularity python.

# Cite

This repository contains code for the paper [http://arxiv.org/abs/2506.00155](http://arxiv.org/abs/2506.00155).
If you find this code useful, please cite the paper:

```
@article{zhan-2025-expres-deter,
  author =	 {Zhan, Ni and Wheeler, William A. and Goldshlager,
                  Gil and Ertekin, Elif and Adams, Ryan P. and Wagner,
                  Lucas K.},
  title =	 {Expressivity of Determinantal Ansatzes for Neural
                  Network Wave Functions},
  journal =	 {CoRR},
  year =	 2025,
  url =		 {http://arxiv.org/abs/2506.00155},
  abstract =	 {},
  archivePrefix ={arXiv},
  eprint =	 {2506.00155},
  primaryClass = {cond-mat.str-el},
}

```