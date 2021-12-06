# Min-Max Adversarial Attacks

[[`Paper`](https://arxiv.org/pdf/1906.03563.pdf)]
[[`arXiv`](https://arxiv.org/abs/1906.03563)]
[[`Poster`](https://github.com/wangjksjtu/minmax-adv/blob/main/imgs/min-max-adv-poster.pdf)]
[[`Slide`](https://neurips.cc/media/neurips-2021/Slides/27929.pdf)]
[[`Project Page`](http://www.cs.toronto.edu/~wangjk/publications/minmax-adv.html)]

> [Adversarial Attack Generation Empowered by Min-Max Optimization]()  
> Jingkang Wang, Tianyun Zhang, Sijia Liu,  Pin-Yu Chen, Jiacen Xu, Makan Fardad, Bo Li \
> NeurIPS 2021  

<div align="center">
    <img src="imgs/revisit-minmax.png" alt><br>
    Revisit the strength of min-max optimization in the context of  adversarial attack generation
</div>

## Reproduce Main Results
Please check `neurips21` folder for reproducing the robust adversarial attack results presented in the paper. We provide detailed instructions in `neurips21/README.md` and bash scripts `neurips21/scripts`.  The code is based on tensorflow 1.x (tested from `1.10.0` - `1.15.0`), which is a bit outdated. Currently, we do not have plans to upgrade it to tensorflow 2.x. If you do not aim to reproduce the exact numbers but use min-max attacks in your projects, we provide a pytorch implementation with latest pre-trained models (e.g., EfficientNet, ViT, etc) and ImageNet-1k supports. Please see the following section for more details.

## Pytorch Implementation
TBD (stay tuned!)

## Citation
If you find our code or paper useful, please consider citing
```bibtex
@inproceedings{wang2021adversarial,
    title={Adversarial Attack Generation Empowered by Min-Max Optimization},
    author={Wang, Jingkang and Zhang, Tianyun and Liu, Sijia  and Chen, Pin-Yu and Xu, Jiacen and Fardad, Makan and Li, Bo},
    booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
    year={2021}
}
```

## Questions/Bugs
Please submit a Github issue or contact wangjk@cs.toronto.edu if you have any questions or find any bugs.
