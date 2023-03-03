# Simple Pooling Front-ends for Efficient Audio Classification

This repository contains the code and models of "[Simple Pooling Front-ends for Efficient Audio Classification
](https://arxiv.org/abs/2210.00943)" [ICASSP 2023].

See usage in main.py:
* frontend: mel-spectrogram computation & temporal dimension reduction (using spectral pooling)
* backbone: CNN-based model for audio feature extraction (pre-trained model is available)
* input: [batch, waveform]
* embedding: [batch, T, 512]

### Citation
```
@inproceedings{liu2022simple,
  title={Simple Pooling Front-ends For Efficient Audio Classification},
  author={Liu, Xubo and Liu, Haohe and Kong, Qiuqiang and Mei, Xinhao and Plumbley, Mark D and Wang, Wenwu},
  booktitle = {IEEE International Conference on Acoustic, Speech and Signal Procssing (ICASSP)},
  year = {2023}
}
```
