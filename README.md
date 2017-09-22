# LeakGAN

## Requirements: 
* **Tensorflow r1.2.1**
* Python 2.7
* CUDA 7.5+ (For GPU)

## Introduction
Apply Generative Adversarial Nets and Hierarchical Reinforcement Learning to generating sequences of discrete tokens using leaked information from D.
![](https://github.com/CR-Gjx/LeakGAN/figures/leakgan.png)
As the illustration of LeakGAN. We specifically introduce a hierarchical generator G, which consists of a high-level MANAGER module and a low-level WORKER module. The MANAGER is a long short term memory network (LSTM) and serves as a mediator. In each step, it receives generator D’s high-level feature representation, e.g., the feature map of the CNN, and uses it to form the guiding goal for the WORKER module in that timestep. As the information from D is internally-maintained and in an adversarial game it is not supposed to provide G with such information. We thus call it a leakage of information from D.
Next, given the goal embedding produced by the MANAGER, the WORKER firstly encodes current generated words with another LSTM, then combines the output of the LSTM and the goal embedding to take a final action at current state. As such, the guiding signals from D are not only available to G at the end in terms of the scalar reward signals, but also available in terms of a goal embedding vector during the generation process to guide G how to get improved.

You can get the code and run the experiments in follow folders.
## Folder

Synthetic Data: synthetic data experiment

Image COCO: a real text example for our model using dataset Image COCO (http://cocodataset.org/#download)

Note: this code is based on the [previous work by LantaoYu](https://github.com/LantaoYu/SeqGAN). Many thanks to [LantaoYu](https://github.com/LantaoYu).
