# Adversarial Attack Generation Empowered by Min-Max Optimization

This is the tensorflow implementation of min-max adversarial attacks as described in the following NeurIPS 2021 paper:

```bibtex
@inproceedings{wang2021adversarial,
    title={Adversarial Attack Generation Empowered by Min-Max Optimization},
    author={Wang, Jingkang and Zhang, Tianyun and Liu, Sijia  and Chen, Pin-Yu and Xu, Jiacen and Fardad, Makan and Li, Bo},
    booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
    year={2021}
}
```

## Installation
### Conda (Recommended)
If you are using conda, you can create a `minmax-adv` environment with all the dependencies by running:

```
git clone https://github.com/wangjksjtu/minmax-adv.git
cd minmax-adv
conda env create -f environment.yaml
source activate minmax-adv
```

### Manual Installation
#### Requirements:
- python 3.5+
- tensorflow 1.15.0
- keras, scipy, pandas, tabulate

```
mkvirtualenv minmax-adv --python=/usr/bin/python3
pip install -r requirements.txt
```

## Quickstarts

#### 0. Download and extract pre-trained models
```
gdrive download 1c1-l2zkv3ooOjU-_Zsmqg8RdNJCLMgp5
tar -xzvf models.tar.gz
```
Note that [gdrive](https://github.com/gdrive-org/gdrive) is a command line utility for interacting with Google Drive, you need to install and configure it properly (see [wiki](https://github.com/gdrive-org/gdrive/wiki) for complete installation). Otherwise, you can manually download the pretrained models [here](https://drive.google.com/file/d/1c1-l2zkv3ooOjU-_Zsmqg8RdNJCLMgp5/view?usp=sharing).

#### 0'. Train your own models (Optional)
If you want to train your own models on benign examples, run `train.py` script and specify the dataset as well as the model architectures in `models.py`. New models can be easily added into `models.py`.
```
python train.py --dataset mnist --model A  # (MLP)
python train.py --dataset mnist --model B  # (All-CNNs)
python train.py --dataset mnist --model C  # (LeNet)
python train.py --dataset mnist --model D  # (LeNetV2)
```
#### 1. Ensemble attack on multiple models
```
python ens_attack.py --avg_case True  --models ABCD   # average opt.
python ens_attack.py --avg_case False --models AEFGJ  # min-max opt.
```

#### 2. Generating universal perturbation to input samples
```
python uni_attack.py --model A --norm 1 --eps 20.0 --avg_case True  # average opt.
python uni_attack.py --model B --norm 2 --eps 3.0 --epochs 50       # min-max opt.
```

#### 3. Robust attack over data transformations
```
python trans_attack.py --model A --K 5  # five transformations in total
```