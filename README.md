Simple Pooling Front-ends (SimPFs) for Efficient Audio Classification

See usage in main.py:
* frontend: mel-spectrogram computation & temporal dimension reduction (using spectral pooling)
* backbone: CNN-based model for audio feature extraction (pre-trained model is available)
* input: [batch, waveform]
* embedding: [batch, T, 512]