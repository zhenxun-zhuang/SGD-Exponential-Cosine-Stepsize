# Exponential Step Sizes for Non-Convex SGD
This repository contains PyTorch codes for the experiments on deep learning in the paper:

**[Exponential Step Sizes for Non-Convex SGD](https://arxiv.org/abs/2002.05273)**  
Xiaoyu Li*, Zhenxun Zhuang*, Francesco Orabona

### Description
Stochastic Gradient Descent (SGD) is a popular tool in training large-scale machine learning models. Its performance, however, is highly variable, depending crucially on the choice of the step sizes. Accordingly, a variety of strategies on tuning the step sizes have been proposed. Yet, most of them lack a theoretical guarantee, whereas those backed by theories often do not shine in practice. Regarding this, we introduce the exponential step sizes, a novel strategy that is simple to use and enjoys both theoretical and empirical support. In particular, we prove its almost optimal convergence rate for stochastic optimization of smooth non-convex functions. Furthermore, in the case where the PL condition holds, this strategy can automatically adapt to the level of noise without knowing it. Finally, we empirically verified on real-world datasets with deep learning architectures that, requiring only two hyperparameters to tune, it bests or matches the performance of various finely-tuned state-of-the-art strategies including Adam and cosine decay.

### Requirements
Run the following command to install required libraries:
```
pip install -r requirements.txt
```
