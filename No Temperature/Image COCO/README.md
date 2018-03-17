# LeakGAN

## Requirements: 
* **Tensorflow r1.2.1**
* Python 2.7
* CUDA 7.5+ (For GPU)

## Introduction
This is the synthetic data experiment of LeakGAN.

## File

LeakGANModel.py : The generator model of LeakGAN including Manager and Worker.

Discriminator.py: The discriminator model of LeakGAN including Feature Extractor and classification.

data_loader.py: Data helpy function for this experiment.

Main.py: The Main function of this experiment.

convert.py: The convert one-hot number to real word.

eval_bleu.py: Evaluation the BLEU scores (2-5) between test datatset and generated data.

## Details 
We provide example codes to repeat the synthetic data experiments with oracle evaluation mechanisms in length 20.
To run the experiment with default parameters for length 20:
```
$ python Main.py
```

In our code, we
The experiment has two stages. In the first stage, use the positive data provided by the oracle and Maximum Likelihood Estimation to perform supervise learning. In the second stage, use adversarial training to improve the generator.

When you running the code, the pre-train model will be store in folder ``ckpts``, if you want to restore the pre-trained discriminator model, you can run:
```
$ python Main.py --resD=True --model=leakgan_preD
``` 

if you want to restore all pre-traine model or unsupervised model (store model every 30 epoch named ``leakgan-31`` or other number), you can run:
```
$ python Main.py --restore=True --model=leakgan_pre
``` 

After running the experiments, you can run the ``convert.py`` to obtain the real sentence in folder ``speech``.You also can run the ``eval_bleu.py`` to acquire the BLEU score in your command line.
The generated examples store in folder ``save`` every 30 epochs, and the file named ``coco_31.txt`` or other numbers.

