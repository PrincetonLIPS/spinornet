FROM nvcr.io/nvidia/jax:24.04-maxtext-py3

RUN pip install ipykernel==6.20.1 \
    && pip install numpy==1.24.2 absl-py==1.4 dm-haiku==0.0.13 ml-collections==0.1.1 \
    && pip install --no-dependencies optax==0.1.5 chex \
    && pip install pandas==1.5.2 pyqmc==0.6.0 pyscf==2.2.1 scipy==1.9.1 tqdm==4.64.1 matplotlib==3.7.1 \
    && pip install attrs==22.2.0 dm-tree==0.1.8 toolz==0.12.0 immutabledict==2.2.4 tensorflow-probability==0.20.1 folx==0.2.11\
    && apt clean && pip cache purge

RUN mkdir /home/spinornet
WORKDIR /home/spinornet

EXPOSE 80