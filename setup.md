# Install
<!-- ```bash
git clone https://github.com/NYCU-MAPL/BridgingCompressionMLLM
cd BridgingCompressionMLLM
```

```bash
conda create -n BridgingCompressionMLLM -y
conda activate BridgingCompressionMLLM
```

```bash
conda install pip -y
pip install -U pip
pip install -e .
pip install git+https://github.com/openai/CLIP.git # this works, btw. 
``` -->

Get this error while doing `pip install -e .`:
  ../meson.build:1:0: ERROR: Compiler /usr/bin/g++-10 cannot compile programs.


Try w/ Python 3.12:

```bash
conda create -n bcmllm python=3.12
conda activate bcmllm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .

pip install git+https://github.com/openai/CLIP.git
```


# Prep

Edit `config/TransformNeck.yaml`:

- `dataset_path` (imagenet dataset)

Download imagenet and the devkit (https://github.com/0429charlie/ImageNet_metadata/blob/master/ILSVRC2012_devkit_t12.tar.gz):

```bash
wget -P /home/ying/datasets/ILSVRC https://github.com/0429charlie/ImageNet_metadata/raw/refs/heads/master/ILSVRC2012_devkit_t12.tar.gz
```

# run 

```bash
cd experiments/mllm/BridgingCompressionMLLM

conda activate bcmllm

python examples/train.py -c config/TransformNeck.yaml
```