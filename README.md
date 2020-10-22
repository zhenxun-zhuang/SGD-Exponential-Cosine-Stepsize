# A Second look at Exponential and Cosine Step Sizes: Simplicity, Convergence, and Performance
This repository contains PyTorch codes for the experiments on deep learning in the paper:

**[Exponential Step Sizes for Non-Convex SGD](https://arxiv.org/abs/2002.05273)**  
Xiaoyu Li*, Zhenxun Zhuang*, Francesco Orabona

### Description
Stochastic Gradient Descent (SGD) is a popular tool in training large-scale machine learning models. Its performance, however, is highly variable, depending crucially on the choice of the step sizes. Accordingly, a variety of strategies for tuning the step sizes have been proposed. Yet, most of them lack a theoretical guarantee, whereas those backed by theories often do not shine in practice. In this paper, we study two heuristic step size schedules whose power has been repeatedly confirmed in practice: the exponential and the cosine step sizes. For the first time, we provide theoretical support for them: we prove their (almost) optimal convergence rates for stochastic optimization of smooth non-convex functions. Furthermore, if in addition, the Polyak-Lojasiewicz (PL) condition holds, they both automatically adapt to the level of noise, with a rate interpolating between a linear rate for the noiseless case and a sub-linear one for the noisy case. Finally, we conduct a fair and comprehensive empirical evaluation of real-world datasets with deep learning architectures. Results show that, even if only requiring at most two hyperparameters to tune, they best or match the performance of various finely-tuned state-of-the-art strategies.

### Requirements
Run the following command to install required libraries:
```
pip install -r requirements.txt
```
