# U-MVS
Official code for ICCV paper "Digging into Uncertainty in Self-supervised Multi-view Stereo" [\[Paper\]](https://openaccess.thecvf.com/content/ICCV2021/html/Xu_Digging_Into_Uncertainty_in_Self-Supervised_Multi-View_Stereo_ICCV_2021_paper.html) [\[Arxiv\]](https://arxiv.org/abs/2108.12966)

## Log

### 2021-12-24

 - The evaulation code of [U-MVS(MVSNet)](./u_mvs_mvsnet/) is released.
 - The training code will be uploaded in a few days.
 - A toy example for understanding the `depth2flow` module is provided in [toy_example_depth2flow](./toy_example_depth2flow/).

## Citation
If you find this work is helpful to your work, please cite:
```
@inproceedings{xu2021digging,
  title={Digging into Uncertainty in Self-supervised Multi-view Stereo},
  author={Xu, Hongbin and Zhou, Zhipeng and Wang, Yali and Kang, Wenxiong and Sun, Baigui and Li, Hao and Qiao, Yu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={6078--6087},
  year={2021}
}
```

## Acknowledgement

The baseline code of this repository is based on [JDACS](https://github.com/ToughStoneX/Self-Supervised-MVS).
We also acknowledge the code of [arflow](https://github.com/lliuz/ARFlow) for their great work in unsupervised flow estimation<sup>[1]</sup>, which is used as the backbone of our RGB2Flow module.
Furthermore, we thank for the Tensorflow implementation in [dl-uncertainty](https://github.com/pmorerio/dl-uncertainty) for aleatoric and epistemic uncertainty estimation<sup>[2]</sup>.

## Reference

[1] L Liu, J Zhang, and etc, "Learning by Analogy: Reliable Supervision from Transformations for Unsupervised Optical Flow Estimation", CVPR 2020

[2] A Kendall, Y Gal, "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?", NIPS 2017
