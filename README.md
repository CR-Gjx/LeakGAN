# LeakGAN
The code of research paper [Long Text Generation via Adversarial Training with Leaked Information](https://arxiv.org/abs/1709.08624). 
## Requirements
* **Tensorflow r1.2.1**
* Python 2.7
* CUDA 7.5+ (For GPU)

## Introduction
Automatically generating coherent and semantically meaningful text has many applications in machine translation, dialogue systems, image captioning, etc. Recently, by combining with policy gradient, Generative Adversarial Nets (GAN) that use a discriminative model to guide the training of the generative model as a reinforcement learning policy has shown promising results in text generation. However, the scalar guiding signal is only available after the entire text has been generated and lacks intermediate information about text structure during the generative process. As such, it limits its success when the length of the generated text samples is long (more than 20 words). In this project, we propose a new framework, called LeakGAN, to address the problem for long text generation. We allow the discriminative net to leak its own high-level extracted features to the generative net to further help the guidance. The generator incorporates such informative signals into all generation steps through an additional Manager module, which takes the extracted features of current generated words and outputs a latent vector to guide the Worker module for next-word generation. Our extensive experiments on synthetic data and various real-world tasks with Turing test demonstrate that LeakGAN is highly effective in long text generation and also improves the performance in short text generation scenarios. More importantly, without any supervision, LeakGAN would be able to implicitly learn sentence structures only through the interaction between Manager and Worker.

![](https://github.com/CR-Gjx/LeakGAN/blob/master/figures/leakgan.png)

As the illustration of LeakGAN. We specifically introduce a hierarchical generator G, which consists of a high-level MANAGER module and a low-level WORKER module. The MANAGER is a long short term memory network (LSTM) and serves as a mediator. In each step, it receives generator D’s high-level feature representation, e.g., the feature map of the CNN, and uses it to form the guiding goal for the WORKER module in that timestep. As the information from D is internally-maintained and in an adversarial game it is not supposed to provide G with such information. We thus call it a leakage of information from D.

Next, given the goal embedding produced by the MANAGER, the WORKER firstly encodes current generated words with another LSTM, then combines the output of the LSTM and the goal embedding to take a final action at current state. As such, the guiding signals from D are not only available to G at the end in terms of the scalar reward signals, but also available in terms of a goal embedding vector during the generation process to guide G how to get improved.

You can get the code and run the experiments in follow folders.
## Folder

Synthetic Data: synthetic data experiment

Image COCO: a real text example for our model using dataset Image COCO (http://cocodataset.org/#download)

Note: this code is based on the [previous work by LantaoYu](https://github.com/LantaoYu/SeqGAN). Many thanks to [LantaoYu](https://github.com/LantaoYu).
