# Explain2Me: Salience-Based Explainability for Synthetic Face Detection Models
<p align="center">
  <img src="https://user-images.githubusercontent.com/30506411/213560601-7bb96cd0-4e17-4a66-bfc1-654442f4790a.png" width="750" height="450" />
</p>

# TL;DR
Model salience is used extensively, but only for a few samples at a time. What about the entire test set? What if we could get more information out of these maps? This repository provides a novel framework to explore salience in a more meaningful way. These metrics could be used with *any* type of salience, whether derived from a neural network (CAM, GradCAM, etc.) or human annotators (eye tracking or written annotations). These methods can be used to decide which model to use, see how the model compares with human intuition, or as a sanity check.

# Paper Details

## Abstract

> Convolutional neural networks have continued to improve their overall performance and accuracy over the last decade. At the same time, while model complexity grows, it has grown increasingly more difficult to explain the model's decisions. Such explanations, beyond mere curiosity, may be of critical importance for a reliable operation of human-machine pairing setups, or when the choice of the best model out of many equally-accurate models must be made. One of the popular ways of explaining the model's decision includes using salience maps to view what regions of an image the model uses to make its prediction. However, examining such salience maps at scale is not practical. In this paper, we propose five novel salience-based metrics that provide an extra (complementary to accuracy) explanation of the model's properties possible to be calculated on large data samples: (a) what is the average model's focus measured by an entropy of its salience, (b) how model's focus changes when fed with out-of-set samples, (c) how model focus follows geometrical transformations, (d) what is the stability of the model's focus in multiple, independent training runs, and (e) how model's focus reacts to salience-guided image degradation. To assess the proposed metrics on a concrete and topical problem, we conducted a series of experiments in a domain of synthetic faca detection and with two types of models: (a) trained traditionally with cross-entropy loss, and (b) guided by human salience when training to increase models' generalization capabilities. These two types of models are characterized by different and interpretable properties of their salience maps, which gives a possibility to evaluate the correctness of the proposed metrics. We offer source codes of the proposed metrics along with this paper.

## Overview of Metrics
![teaser_v3 (1)-1](https://user-images.githubusercontent.com/30506411/213560634-4083952a-c20b-4056-8833-9dd42b450f03.png)

### Salience Entropy
Explainable Models typically have focused salience on the object it is classifying. High accuracy scores with unfocused salience maps may be an indication that the model is overfitting, or classifying images by poor features.

### Salience-Assessed Reaction to Noise
![saltpepper-figure-resnet](https://user-images.githubusercontent.com/30506411/215517767-3a1b72dc-42e9-4fbf-a579-7ae965940518.png)

Explainable Models should degrade their performance gracefully in presence of noise. 

### Salience Resilience to Geometrical Transformations
<p align="center">
  <img src="https://user-images.githubusercontent.com/30506411/213561467-672e32c9-8339-4f6c-a1d0-7ae69e93fa49.png" width="800" height="600" />
</p>

Explainable models should be able to handle basic geometrical transformations, e.g. the image is upsidedown, or rotated sideways.

### Salience-Based Image Degradation
<p align="center">
  <img src="https://user-images.githubusercontent.com/30506411/213559873-44487b73-6456-4d20-9c1c-5ccc90090816.png" width="750" height="450" />
</p>
If salient regions of the model are important in the model's decision making process, then removing them from an input image should negatively affect the model's ability to classify the images. Conversely, the removal of non-salient regions of an input image should not significantly affect the model's performance.

### Salience Stability Across Training Runs
![stability-figure-resnet](https://user-images.githubusercontent.com/30506411/213558890-0731925e-25dc-4aa9-bbc3-e1ed61c3caa6.png)

Models with the same backbone architecture, training parameters, and training data should have consistent salience maps.


## Citation

# How To Use
The model weights and the entire project can be downloaded via [Google Drive](https://drive.google.com/drive/folders/124MVxHG1nVj9dxY0cnnAMiL8UpS9MKJ5?usp=share_link).



# Resources
[TorchCAM](https://frgfm.github.io/torch-cam/)
