# Action Recognition on on Kinetics-400 Subset

This instructions to **reproduce the results** of VideoMAE V2 on a **subset of the Kinetics-400 dataset**, focusing on selected action categories:

- **Dance-related**: `dancing ballet`, `breakdancing`, `salsa dancing`, `swing dancing`, `zumba`
- **Sport-related**: `playing badminton`, `playing basketball`, `playing ice hockey`, `playing tennis`, `playing volleyball`


## ✅ Assumptions

- The subset of Kinetics-400 has been **prepared and placed in `./data`**, following the structure described in [DATASET.md](docs/DATASET.md).
- You are using the **pretrained model checkpoint** located at:

```bash
https://drive.google.com/file/d/1VT0hnxVlKuy32N-SSbL-JkfHZgX3YJiB/view
```
## Step 1. Fine-tune on Subset
To fine-tune the pretrained model on the subset:
```bash
bash finetune-videomaev2.sh
```
The best model path you can find at 
```bash
https://drive.google.com/file/d/1C9t0UcK4WeeHvz7nnyHzLEH3I8whiHf3/view
```

## Step 2: Evaluate on Subset
To evaluate the fine-tuned model:
```bash
bash test-videomaev2.sh
```


---
## [CVPR 2023] Official Implementation of VideoMAE V2

![flowchart](misc/VideoMAEv2_flowchart.png)

> [**VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking**](https://arxiv.org/abs/2303.16727)<br>
> [Limin Wang](http://wanglimin.github.io/), [Bingkun Huang](https://github.com/congee524), [Zhiyu Zhao](https://github.com/JerryFlymi), [Zhan Tong](https://scholar.google.com/citations?user=6FsgWBMAAAAJ), [Yinan He](https://dblp.org/pid/93/7763.html), [Yi Wang](https://scholar.google.com.hk/citations?hl=zh-CN&user=Xm2M8UwAAAAJ), [Yali Wang](https://scholar.google.com/citations?user=hD948dkAAAAJ), and [Yu Qiao](https://scholar.google.com/citations?user=gFtI-8QAAAAJ&hl)<br>
> Nanjing University, Shanghai AI Lab, CAS<br>

### News
**[2024.09.19]** Checkpoints have been migrated to Hugging Face. You can obtain weights from [VideoMAEv2-hf](https://huggingface.co/OpenGVLab/VideoMAE2/tree/main).<br>
**[2023.05.29]** VideoMAE V2-g features for THUMOS14 and FineAction datasets are available at [TAD.md](docs/TAD.md) now.<br>
**[2023.05.11]** We have supported testing of our distilled models at MMAction2 (dev version)! See [PR#2460](https://github.com/open-mmlab/mmaction2/pull/2460).<br>
**[2023.05.11]** The feature extraction script for TAD datasets has been released! See instructions at [TAD.md](docs/TAD.md).<br>
**[2023.04.19]** ViT-giant model weights have been released! You can get the download links from [MODEL_ZOO.md](docs/MODEL_ZOO.md).<br>
**[2023.04.18]** Code and the distilled models (vit-s & vit-b) have been released!<br>
**[2023.04.03]** ~~Code and models will be released soon.~~<br>


### Model Zoo

We now provide the model weights in [MODEL_ZOO.md](docs/MODEL_ZOO.md). We have additionally provided distilled models in MODEL_ZOO.

|  Model  | Dataset | Teacher Model | \#Frame | K710 Top-1 | K400 Top-1 | K600 Top-1 |
| :-----: | :-----: | :-----------: | :-----: | :--------: | :--------: | :--------: |
| ViT-small | K710 | vit_g_hybrid_pt_1200e_k710_ft | 16x5x3 | 77.6 | 83.7 | 83.1 |
| ViT-base | K710 | vit_g_hybrid_pt_1200e_k710_ft | 16x5x3 | 81.5 | 86.6 | 85.9 |

### Installation

Please follow the instructions in [INSTALL.md](docs/INSTALL.md).

### Data Preparation

Please follow the instructions in [DATASET.md](docs/DATASET.md) for data preparation.

### Pre-training

The pre-training instruction is in [PRETRAIN.md](docs/PRETRAIN.md).

### Fine-tuning

The fine-tuning instruction is in [FINETUNE.md](docs/FINETUNE.md).

### Citation

If you find this repository useful, please use the following BibTeX entry for citation.

```latex
@InProceedings{wang2023videomaev2,
    author    = {Wang, Limin and Huang, Bingkun and Zhao, Zhiyu and Tong, Zhan and He, Yinan and Wang, Yi and Wang, Yali and Qiao, Yu},
    title     = {VideoMAE V2: Scaling Video Masked Autoencoders With Dual Masking},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {14549-14560}
}

@misc{videomaev2,
      title={VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking},
      author={Limin Wang and Bingkun Huang and Zhiyu Zhao and Zhan Tong and Yinan He and Yi Wang and Yali Wang and Yu Qiao},
      year={2023},
      eprint={2303.16727},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
