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

target_lstm.py : The oracle model for length 40.
target_lstm20.py : The oracle model for length 20.

data_loader.py: Data helpy function for this experiment.

Main.py: The Main function of this experiment.

## Details 
We provide example codes to repeat the synthetic data experiments with oracle evaluation mechanisms in length 20 and 40.
To run the experiment with default parameters for length 20:
```
$ python Main.py
```

To run the experiment with default parameters for length 40:
```
$ python Main.py --length=40
```

In our code, we
The experiment has two stages. In the first stage, use the positive data provided by the oracle and Maximum Likelihood Estimation to perform supervise learning. In the second stage, use adversarial training to improve the generator.

When you running the code, the pre-train model will be store in folder ``ckpts``, if you want to restore the pre-trained discriminator model, you can run:
```
$ python Main.py --resD=True --model=leakgan_preD
``` 

if you want to restore all pre-trained model, you can run:
```
$ python Main.py --restore=True --model=leakgan_pre
``` 

After running the experiments, you could get the negative log-likelihodd performance printed in the console like:
#### Length 40:
```
pre-train epoch  0 test_loss  12.26
Groud-Truth: 4.18953
pre-train epoch  5 test_loss  11.5891
Groud-Truth: 4.18663
Start pre-training...
pre-train epoch  0 test_loss  11.0077
Groud-Truth: 4.18508
pre-train epoch  5 test_loss  10.1536
Groud-Truth: 4.18771
Start pre-training...
pre-train epoch  0 test_loss  9.77986
Groud-Truth: 4.17996
pre-train epoch  5 test_loss  9.38056
Groud-Truth: 4.18065
Start pre-training...
pre-train epoch  0 test_loss  9.27072
Groud-Truth: 4.18544
pre-train epoch  5 test_loss  9.21736
Groud-Truth: 4.1801
Start pre-training...
pre-train epoch  0 test_loss  9.1791
Groud-Truth: 4.18622
pre-train epoch  5 test_loss  9.1613
Groud-Truth: 4.17681
Start pre-training...
pre-train epoch  0 test_loss  9.11273
Groud-Truth: 4.18173
pre-train epoch  5 test_loss  9.11366
Groud-Truth: 4.18395
Start pre-training...
pre-train epoch  0 test_loss  9.09866
Groud-Truth: 4.18976
pre-train epoch  5 test_loss  9.12016
Groud-Truth: 4.18625
Start pre-training...
pre-train epoch  0 test_loss  9.14861
Groud-Truth: 4.17891
^[[Apre-train epoch  5 test_loss  9.18029
Groud-Truth: 4.18918
Start pre-training...
pre-train epoch  0 test_loss  9.20815
Groud-Truth: 4.1869
pre-train epoch  5 test_loss  9.24535
Groud-Truth: 4.1824
Start pre-training...
pre-train epoch  0 test_loss  9.25886
Groud-Truth: 4.18081
pre-train epoch  5 test_loss  9.29524
Groud-Truth: 4.18282
#########################################################################
Start Adversarial Training...
total_batch:  0    -0.0652316    2.3433
total_batch:  0 test_loss:  9.2945
Groud-Truth: 4.18135
total_batch:  1    -0.064719    2.31054
total_batch:  2    -0.063226    2.27155
total_batch:  3    -0.0639394    2.26772
total_batch:  4    -0.0623998    2.21335
total_batch:  5    -0.0625717    2.14454
total_batch:  5 test_loss:  8.89057
Groud-Truth: 4.18198
total_batch:  6    -0.0586181    2.12369
total_batch:  7    -0.0611555    2.11141
total_batch:  8    -0.0570103    2.03449
total_batch:  9    -0.0616121    2.06209
total_batch:  10    -0.0517034    2.02423
total_batch:  10 test_loss:  8.65749
Groud-Truth: 4.18458
total_batch:  11    -0.0446114    1.98766
total_batch:  12    -0.0468828    1.99969
total_batch:  13    -0.0428705    1.98661
total_batch:  14    -0.0359654    1.98278
total_batch:  15    -0.043911    1.93812
total_batch:  15 test_loss:  8.41053
Groud-Truth: 4.1857
total_batch:  16    -0.0425446    1.91711
total_batch:  17    -0.0359852    1.89312
total_batch:  18    -0.0273773    1.88675
total_batch:  19    -0.0391433    1.87262
total_batch:  20    -0.0350945    1.84019
total_batch:  20 test_loss:  8.27498
Groud-Truth: 4.18312
total_batch:  21    -0.0370521    1.87387
total_batch:  22    -0.033906    1.82186
total_batch:  23    -0.0354352    1.81875
total_batch:  24    -0.0320315    1.84671
total_batch:  25    -0.0351358    1.83876
total_batch:  25 test_loss:  8.10912
Groud-Truth: 4.18632
total_batch:  26    -0.0368386    1.83973
total_batch:  27    -0.0361074    1.81938
total_batch:  28    -0.032679    1.78491
total_batch:  29    -0.0355964    1.79323
total_batch:  30    -0.0341959    1.80316
total_batch:  30 test_loss:  8.03633
Groud-Truth: 4.18646
total_batch:  31    -0.0361932    1.79637
total_batch:  32    -0.0335791    1.75044
total_batch:  33    -0.0345367    1.79666
total_batch:  34    -0.0354369    1.73919
total_batch:  35    -0.0357523    1.72717
total_batch:  35 test_loss:  7.90693
Groud-Truth: 4.18183
total_batch:  36    -0.0356606    1.67951
total_batch:  37    -0.0357596    1.73858
total_batch:  38    -0.032771    1.65516
total_batch:  39    -0.0354646    1.65092
total_batch:  40    -0.0326444    1.67265
total_batch:  40 test_loss:  7.79181
Groud-Truth: 4.18917
total_batch:  41    -0.0332658    1.66889
total_batch:  42    -0.034617    1.64287
total_batch:  43    -0.0329468    1.67488
total_batch:  44    -0.0353837    1.63027
total_batch:  45    -0.0318517    1.58585
total_batch:  45 test_loss:  7.63169
Groud-Truth: 4.17742
total_batch:  46    -0.0348847    1.57419
total_batch:  47    -0.0330474    1.57364
total_batch:  48    -0.0297963    1.55306
total_batch:  49    -0.0328465    1.54369
total_batch:  50    -0.0328482    1.53558
total_batch:  50 test_loss:  7.53855
Groud-Truth: 4.17784
total_batch:  51    -0.0306785    1.50616
total_batch:  52    -0.0310525    1.52157
total_batch:  53    -0.0331497    1.50733
total_batch:  54    -0.031042    1.45315
total_batch:  55    -0.0298762    1.45128
total_batch:  55 test_loss:  7.39215
Groud-Truth: 4.17628
total_batch:  56    -0.0283417    1.39843
total_batch:  57    -0.0282539    1.39738
total_batch:  58    -0.0274459    1.3812
total_batch:  59    -0.0289517    1.37688
total_batch:  60    -0.0256093    1.4402
total_batch:  60 test_loss:  7.41288
Groud-Truth: 4.18367
total_batch:  61    -0.0311947    1.43996
total_batch:  62    -0.02923    1.36832
total_batch:  63    -0.0267876    1.38198
total_batch:  64    -0.0305385    1.39321
total_batch:  65    -0.0312742    1.45848
total_batch:  65 test_loss:  7.3914
Groud-Truth: 4.18158
total_batch:  66    -0.029835    1.40807
total_batch:  67    -0.0305667    1.32871
total_batch:  68    -0.0303886    1.40174
total_batch:  69    -0.0277299    1.40849
total_batch:  70    -0.0289449    1.3429
total_batch:  70 test_loss:  7.31235
Groud-Truth: 4.18048
total_batch:  71    -0.0258161    1.32971
total_batch:  72    -0.0248826    1.26198
total_batch:  73    -0.0244172    1.26416
total_batch:  74    -0.0269533    1.2729
total_batch:  75    -0.0251514    1.28549
total_batch:  75 test_loss:  7.30436
Groud-Truth: 4.18839
total_batch:  76    -0.0236495    1.25146
total_batch:  77    -0.0230818    1.2424
total_batch:  78    -0.0256981    1.22673
total_batch:  79    -0.0215024    1.25993
total_batch:  80    -0.023252    1.21244
total_batch:  80 test_loss:  7.16439
Groud-Truth: 4.18141
total_batch:  81    -0.0276772    1.18596
total_batch:  82    -0.0270498    1.22264
```

#### Length 20:
```
pre-train epoch  0 test_loss  9.69433
Groud-Truth: 5.75101
pre-train epoch  5 test_loss  9.03316
Groud-Truth: 5.75479
Start pre-training...
pre-train epoch  0 test_loss  8.58167
Groud-Truth: 5.75434
pre-train epoch  5 test_loss  8.21973
Groud-Truth: 5.7579
Start pre-training...
pre-train epoch  0 test_loss  8.13603
Groud-Truth: 5.75325
pre-train epoch  5 test_loss  8.15756
Groud-Truth: 5.74576
Start pre-training...
pre-train epoch  0 test_loss  8.17478
Groud-Truth: 5.75582
pre-train epoch  5 test_loss  8.16642
Groud-Truth: 5.75338
Start pre-training...
pre-train epoch  0 test_loss  8.15797
Groud-Truth: 5.7517
pre-train epoch  5 test_loss  8.2205
Groud-Truth: 5.75411
Start pre-training...
pre-train epoch  0 test_loss  8.19299
Groud-Truth: 5.74077
pre-train epoch  5 test_loss  8.23519
Groud-Truth: 5.75896
Start pre-training...
pre-train epoch  0 test_loss  8.23879
Groud-Truth: 5.7437
pre-train epoch  5 test_loss  8.28305
Groud-Truth: 5.74944
Start pre-training...
pre-train epoch  0 test_loss  8.29373
Groud-Truth: 5.74839
pre-train epoch  5 test_loss  8.31093
Groud-Truth: 5.74664
Start pre-training...
pre-train epoch  0 test_loss  8.27784
Groud-Truth: 5.76347
pre-train epoch  5 test_loss  8.30236
Groud-Truth: 5.75073
Start pre-training...
pre-train epoch  0 test_loss  8.35919
Groud-Truth: 5.74792
pre-train epoch  5 test_loss  8.34681
Groud-Truth: 5.74609
#########################################################################
Start Adversarial Training...
total_batch:  0    -0.0786467    2.6426
total_batch:  0 test_loss:  8.24163
Groud-Truth: 5.75979
total_batch:  1    -0.0753347    2.53958
total_batch:  2    -0.0752082    2.46282
total_batch:  3    -0.0743724    2.37566
total_batch:  4    -0.0738771    2.36256
total_batch:  5    -0.0723427    2.27869
total_batch:  5 test_loss:  7.8844
Groud-Truth: 5.74654
total_batch:  6    -0.072562    2.35351
total_batch:  7    -0.0721927    2.30898
total_batch:  8    -0.06853    2.14714
total_batch:  9    -0.0674056    2.15058
total_batch:  10    -0.0660952    2.08341
total_batch:  10 test_loss:  7.57832
Groud-Truth: 5.76818
total_batch:  11    -0.0612625    2.01803
total_batch:  12    -0.0583717    1.88487
total_batch:  13    -0.0581408    2.02243
total_batch:  14    -0.0568922    1.91171
total_batch:  15    -0.0457503    1.90713
total_batch:  15 test_loss:  7.42915
Groud-Truth: 5.74912
total_batch:  16    -0.0511991    1.87393
total_batch:  17    -0.0468395    1.79838
total_batch:  18    -0.0524906    1.65953
total_batch:  19    -0.0423182    1.6548
total_batch:  20    -0.0395193    1.67308
total_batch:  20 test_loss:  7.16191
Groud-Truth: 5.75412
total_batch:  21    -0.0489724    1.61477
total_batch:  22    -0.0514075    1.61549
total_batch:  23    -0.039316    1.69423
total_batch:  24    -0.0437855    1.60816
total_batch:  25    -0.0460445    1.5999
total_batch:  25 test_loss:  6.93066
Groud-Truth: 5.76414
total_batch:  26    -0.0390215    1.53606
total_batch:  27    -0.0370758    1.49105
total_batch:  28    -0.040418    1.48834
total_batch:  29    -0.0370537    1.53179
total_batch:  30    -0.0400711    1.50991
total_batch:  30 test_loss:  6.87948
Groud-Truth: 5.75843
total_batch:  31    -0.0393979    1.50215
total_batch:  32    -0.0361043    1.36426
total_batch:  33    -0.0412544    1.43601
total_batch:  34    -0.0366505    1.4642
total_batch:  35    -0.0404557    1.41566
total_batch:  35 test_loss:  6.75794
Groud-Truth: 5.75188
total_batch:  36    -0.0432183    1.45198
total_batch:  37    -0.0393205    1.40172
total_batch:  38    -0.0403796    1.29743
total_batch:  39    -0.0352011    1.49704
total_batch:  40    -0.0433412    1.46439
total_batch:  40 test_loss:  6.89491
Groud-Truth: 5.74032
total_batch:  41    -0.0399952    1.43648
total_batch:  42    -0.0421722    1.40709
total_batch:  43    -0.04585    1.45173
total_batch:  44    -0.0444828    1.35187
total_batch:  45    -0.0437096    1.36157
total_batch:  45 test_loss:  6.71096
Groud-Truth: 5.75132
total_batch:  46    -0.0393726    1.42049
total_batch:  47    -0.0362047    1.30559
total_batch:  48    -0.0344743    1.32346
total_batch:  49    -0.038387    1.28226
total_batch:  50    -0.0374189    1.37834
total_batch:  50 test_loss:  6.84156
Groud-Truth: 5.76275
total_batch:  51    -0.0391664    1.33567
total_batch:  52    -0.0407602    1.39949
total_batch:  53    -0.0370343    1.37453
total_batch:  54    -0.04027    1.32105
total_batch:  55    -0.0423864    1.40343
total_batch:  55 test_loss:  6.89914
Groud-Truth: 5.75197
total_batch:  56    -0.0392567    1.37611
total_batch:  57    -0.0432047    1.40969
total_batch:  58    -0.0450205    1.34412
total_batch:  59    -0.0330139    1.35893
total_batch:  60    -0.0356952    1.38876
total_batch:  60 test_loss:  6.94015
Groud-Truth: 5.75518
total_batch:  61    -0.0326097    1.28673
total_batch:  62    -0.0342327    1.36097
total_batch:  63    -0.0375812    1.35249
total_batch:  64    -0.0333246    1.30329
total_batch:  65    -0.0491887    1.40754
total_batch:  65 test_loss:  7.04084
Groud-Truth: 5.74344
total_batch:  66    -0.0502178    1.48692
total_batch:  67    -0.0371131    1.29103
total_batch:  68    -0.0395989    1.34005
total_batch:  69    -0.0411862    1.39341
total_batch:  70    -0.0406015    1.39009
total_batch:  70 test_loss:  7.02006
Groud-Truth: 5.75325
total_batch:  71    -0.0418528    1.39724
total_batch:  72    -0.0436076    1.38301
total_batch:  73    -0.0379975    1.38195
total_batch:  74    -0.0480139    1.3805
total_batch:  75    -0.0434777    1.40051
total_batch:  75 test_loss:  7.01042
Groud-Truth: 5.74453
total_batch:  76    -0.0350107    1.33029
total_batch:  77    -0.035438    1.39518
total_batch:  78    -0.0354185    1.40636
total_batch:  79    -0.0425667    1.39818
total_batch:  80    -0.0332933    1.3926
total_batch:  80 test_loss:  7.07472
Groud-Truth: 5.75491
total_batch:  81    -0.0500576    1.40246
total_batch:  82    -0.0318528    1.36058
total_batch:  83    -0.052383    1.35554
total_batch:  84    -0.0392252    1.40638
total_batch:  85    -0.0392072    1.3252
total_batch:  85 test_loss:  7.01408
Groud-Truth: 5.75238
total_batch:  86    -0.0370223    1.38558
total_batch:  87    -0.038433    1.46076
total_batch:  88    -0.0341441    1.34835
total_batch:  89    -0.0357629    1.31941
total_batch:  90    -0.0339176    1.33506
total_batch:  90 test_loss:  7.01058
Groud-Truth: 5.75113
total_batch:  91    -0.0315428    1.3102
```

