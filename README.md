# FLM
Official code for "Accelerating Vision-Language Pretraining with Free Language Modeling" (CVPR 2023) 

Paper: https://arxiv.org/abs/2303.14038


## Introduction


![](imgs/LMs.png)
The state of the arts in vision-language pretraining (VLP) achieves exemplary performance but suffers from high training costs resulting from slow convergence and long training time, especially on large-scale web datasets. An essential obstacle to training efficiency lies in the entangled prediction rate (percentage of tokens for reconstruction) and corruption rate (percentage of corrupted tokens) in masked language modeling (MLM), that is, a proper corruption rate is achieved at the cost of a large portion of output tokens being excluded from prediction loss. 

Free language modeling (FLM) is a new language modeling method that enables a 100% prediction rate with arbitrary corruption rates. FLM successfully frees the prediction rate from the tie-up with the corruption rate while allowing the corruption spans to be customized for each token to be predicted. FLM-trained models are encouraged to learn better and faster given the same GPU time by exploiting bidirectional contexts more flexibly. 
<!-- ![](imgs/pipeline.png) -->
<p align="center">
  <img src="imgs/pipeline.png" width = "50%" />
</p>

## Pretraining

Coming soon.

## Citation
```
@misc{wang2023accelerating,
      title={Accelerating Vision-Language Pretraining with Free Language Modeling}, 
      author={Teng Wang and Yixiao Ge and Feng Zheng and Ran Cheng and Ying Shan and Xiaohu Qie and Ping Luo},
      year={2023},
      eprint={2303.14038},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```