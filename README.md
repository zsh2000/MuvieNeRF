> # [ICCV 2023] Multi-task View Synthesis with Neural Radiance Fields <br>
> [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Zheng_Multi-task_View_Synthesis_with_Neural_Radiance_Fields_ICCV_2023_paper.pdf)
> [Website](https://zsh2000.github.io/mtvs.github.io/)

This repository contains a PyTorch implementation of our paper "Multi-task View Synthesis with Neural Radiance Fields".

## Installation

#### Tested on a single NVIDIA A100 GPU with 40GB memory.

To install the dependencies, follow the official repository of [GeoNeRF](https://github.com/idiap/GeoNeRF):

## Dataset

TODO.

## Training

Our training process contains two stages following [cross-stitch networks](https://openaccess.thecvf.com/content_cvpr_2016/papers/Misra_Cross-Stitch_Networks_for_CVPR_2016_paper.pdf). In the first stage, we train all the parameters except for the self-attention modules in the CTA module for 5,000 iterations. Run the command:

```
bash train_first_stage_scripts.sh
```

Afterwards, before proceeding to the second stage, change the path of the loaded pretrained weight on L764 in `run_geo_nerf.py`. Then run the command:

```
bash train_second_stage_scripts.sh
```


## Testing

Before testing, change the path of the loaded pretrained weight on L778. Running the following scripts will start testing the multi-task view synthesis task 
on three novel scenes. 

```
bash replica_test_scripts.sh
```

## Citation
If you find our work useful, please consider citing:
```BibTeX
@inproceedings{zheng2023mtvs,
  title={Multi-task View Synthesis with Neural Radiance Fields},
  author={Zheng, Shuhong and Bao, Zhipeng and Hebert, Martial and Wang, Yu-Xiong},
  booktitle={IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}
```

### Acknowledgement
The codes are largely borrowed from the PyTorch implementation of GeoNeRF:

https://github.com/idiap/GeoNeRF

This work was supported in part by NSF Grant 2106825, Toyota Research Institute, NIFA Award 2020-67021-32799, the Jump ARCHES endowment, the NCSA Fellows program, the Illinois-Insper Partnership, and the Amazon Research Award. This work used NVIDIA GPUs at NCSA Delta through allocations CIS220014 and CIS230012 from the ACCESS program.