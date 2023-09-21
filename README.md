# Delving into Crispness: Guided Label Refinement for Crisp Edge Detection

Official implementation of the paper:

Delving into Crispness: Guided Label Refinement for Crisp Edge Detection
[[arXiv](https://arxiv.org/abs/2306.15172)] [[Video](https://www.bilibili.com/video/BV1rj411S7WK)].

[Yunfan Ye](https://yunfan1202.github.io), 
[Renjiao Yi](https://renjiaoyi.github.io/), 
[Zhirui Gao](), 
[Zhiping Cai](), 
[Kai Xu](http://kevinkaixu.net/index.html).

## Changelog 

* [July 2023] Initial release of code and models.


## Abstract

Learning-based edge detection usually suffers from predicting thick edges. Through extensive quantitative study with a new edge crispness measure, we find that noisy human-labeled edges are the main cause of thick predictions. Based on this observation, we advocate that more attention should be paid on label quality than on model design to achieve crisp edge detection. To this end, we propose an effective Canny-guided refinement of human-labeled edges whose result can be used to train crisp edge detectors. Essentially, it seeks for a subset of over-detected Canny edges that best align human labels. We show that several existing edge detectors can be turned into a crisp edge detector through training on our refined edge maps. Experiments demonstrate that deep models trained with refined edges achieve significant performance boost of crispness from 17.4% to 30.6%. With the PiDiNet backbone, our method improves ODS and OIS by 12.2% and 12.6% on the Multicue dataset, respectively, without relying on non-maximal suppression. We further conduct experiments and show the superiority of our crisp edge detection for optical flow estimation and image segmentation.

![](./figures/teaser.png)

## Enviroments
This code has been tested with Ubuntu 18.04, one 3080Ti GPU with CUDA 11.4, Python 3.8, Pytorch 1.12 and Matlab 2019a.

Ealier versions may also work.

## Usage

To refine human-annotated labels, simply run (example):
```bash
python refine_label.py
```
The [EdgeModel_gen.pth](https://drive.google.com/drive/folders/1KyXz4W4SAvfsGh3NJ7XgdOv5t46o-8aa) and [networks.py](https://github.com/knazeri/edge-connect/blob/master/src/networks.py) are borrowed from [EdgeConnect](https://github.com/knazeri/edge-connect).

To evaluate the proposed new metric `Crispness` of predicted edge maps , see `eval_crispness/eval_crispness.m` and run with Matlab.

To evaluate ODS and OIS, see [Structured Edge Detection Toolbox](https://github.com/pdollar/edges) for more details.

To test with pre-trained PiDiNet model, simply run:
```bash
python pidinet/test_pidinet.py
```
The model named `BSDS_refine_dice.pth` is trained on BSDS dataset using refined labels and dice loss.
(using both dice loss and refined labels are highly recommanded for crisp edge detection).

## Citation
```bibtex
@article{ye2023delving,
  title={Delving into Crispness: Guided Label Refinement for Crisp Edge Detection},
  author={Ye, Yunfan and Yi, Renjiao and Gao, Zhirui and Cai, Zhiping and Xu, Kai},
  journal={IEEE Transactions on Image Processing},
  year={2023},
  publisher={IEEE}
}
```

## Acknowledgements

- [PiDiNet](https://github.com/hellozhuo/pidinet)
- [DexiNed](https://github.com/xavysp/DexiNed)
- [LPCB](https://arxiv.org/abs/1807.10097) (Dice loss)
- [CATS](https://github.com/WHUHLX/CATS) (tracing loss)
- [EpicFlow](http://lear.inrialpes.fr/src/epicflow/)
