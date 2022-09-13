Simple Pooling Front-ends (SimPFs) for Efficient Audio Classification

See useage in main.py:
* frontend: mel-spectrogram & temporal dimension reduction (using spectral pooling)
* backbone: CNN-based model for audio feature extraction (pre-trained model is available)
* input: [batch, waveform]
* embedding: [batch, T, 512]