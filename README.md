U-Net speech enhancement
====

## Overview
Implementation of a speech enhancement algorithm using the U-Net in frequency domain [1].

The "U-Net_speech_enhancement.py" has been implemented for speech enhancement, using the encoder-decoder-type architecture known as the U-Net [2]. The algorithm includes two steps (mode), namely training and evaluation step. Once training a U-Net model in the first step, you can apply the pre-learned model to any noisy speech signal.

## Requirement
soundfile 0.10.3

matplotlib 3.1.0

h5py 2.8.0

numpy 1.18.1

scipy 1.4.1

tensorflow-gpu 2.1.0

scikit-learn 0.21.2

pystoi 0.3.3 (only for evaluation metrics)

pesq 0.0.1 (only for evaluation metrics)


## Dataset preparation
Please create a "NOISY" and "CLEAN" foler in the "audio_data" directory, and put speech files (.wav format) in it. The original paper used the noisy speech database provided in [3].


## References
[1] A. E. Bulut and K. Koishida: 'Low-Latency Single Channel Speech Enhancement Using U-Net Convolutional Neural Networks', IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), (2020)

[2] O. Ronneberger, P. Fischer, T. Brox: 'U-Net: Convolutional Networks for Biomedical Image Segmentation', in Proceedings of the International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI), pp.234–241, (2015)

[3] C. Valentini-Botinhao: 'Noisy Speech Database for Training Speech Enhancement Algorithms and TTS models', [Online], Available: http://dx.doi.org/10.7488/ds/2117, (2017)