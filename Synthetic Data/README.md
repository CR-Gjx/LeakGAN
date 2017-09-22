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

target_lstm.py : The oracle model for length 20.
target_lstm40.py : The oracle model for length 40.

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
```
#### Length 40:
pre-train epoch  0 test_loss  11.8427
Groud-Truth: 4.07737
pre-train epoch  5 test_loss  11.246
Groud-Truth: 4.07743
pre-train epoch  10 test_loss  10.0818
Groud-Truth: 4.07343
pre-train epoch  15 test_loss  9.51961
Groud-Truth: 4.06427
pre-train epoch  20 test_loss  9.21217
Groud-Truth: 4.07266
pre-train epoch  25 test_loss  8.96886
Groud-Truth: 4.06702
pre-train epoch  30 test_loss  8.7972
Groud-Truth: 4.07053
pre-train epoch  35 test_loss  8.70575
Groud-Truth: 4.07588
pre-train epoch  40 test_loss  8.74605
Groud-Truth: 4.07283
pre-train epoch  45 test_loss  8.78936
Groud-Truth: 4.07124
pre-train epoch  50 test_loss  8.65296
Groud-Truth: 4.07492
pre-train epoch  55 test_loss  8.87488
Groud-Truth: 4.07114
pre-train epoch  60 test_loss  8.55235
Groud-Truth: 4.07503
pre-train epoch  65 test_loss  8.81821
Groud-Truth: 4.07357
pre-train epoch  70 test_loss  8.75773
Groud-Truth: 4.07609
pre-train epoch  75 test_loss  8.89709
#########################################################################
Start Adversarial Training...
otal_batch:  0 test_loss:  9.20067
Groud-Truth: 4.06864
total_batch:  1    -0.0103204    0.432232
total_batch:  2    -0.00635825    0.248288
total_batch:  3    -0.00608721    0.235681
total_batch:  4    -0.00351167    0.16841
total_batch:  5    -0.00444894    0.203372
total_batch:  5 test_loss:  8.882
Groud-Truth: 4.07084
total_batch:  6    -0.00522597    0.209648
total_batch:  7    -0.00384385    0.173008
total_batch:  8    -0.00450124    0.188541
total_batch:  9    -0.005966    0.214622
total_batch:  10    -0.00439055    0.174395
total_batch:  10 test_loss:  8.67052
Groud-Truth: 4.06625
total_batch:  11    -0.00455342    0.165933
total_batch:  12    -0.00450017    0.160548
total_batch:  13    -0.00469874    0.172401
total_batch:  14    -0.00483039    0.170497
total_batch:  15    -0.00486607    0.163429
total_batch:  15 test_loss:  8.13268
Groud-Truth: 4.07326
total_batch:  16    -0.00447564    0.148062
total_batch:  17    -0.00506889    0.16043
total_batch:  18    -0.00500978    0.149576
total_batch:  19    -0.00495353    0.155296
total_batch:  20    -0.00412809    0.136845
total_batch:  20 test_loss:  8.23201
Groud-Truth: 4.07694
total_batch:  21    -0.00449483    0.147475
total_batch:  22    -0.00554714    0.171629
total_batch:  23    -0.00602938    0.180452
total_batch:  24    -0.00810114    0.217036
total_batch:  25    -0.00827135    0.22581
total_batch:  25 test_loss:  7.31975
Groud-Truth: 4.07192
total_batch:  26    -0.00788617    0.219955
total_batch:  27    -0.00910377    0.248353
total_batch:  28    -0.0101237    0.272106
total_batch:  29    -0.0113172    0.30652
total_batch:  30    -0.0103402    0.281662
total_batch:  30 test_loss:  7.30227
Groud-Truth: 4.07328
total_batch:  31    -0.0111794    0.28898
total_batch:  32    -0.00966522    0.258767
total_batch:  33    -0.00928732    0.246102
total_batch:  34    -0.00927821    0.242015
total_batch:  35    -0.00788258    0.215419
total_batch:  35 test_loss:  7.23739
Groud-Truth: 4.07588
total_batch:  36    -0.00708662    0.190525
total_batch:  37    -0.00746638    0.202252
total_batch:  38    -0.00936851    0.253638
total_batch:  39    -0.00905509    0.232895
total_batch:  40    -0.00767142    0.203772
total_batch:  40 test_loss:  7.19115
Groud-Truth: 4.07375
total_batch:  41    -0.00868896    0.218709
total_batch:  42    -0.0108744    0.264243
total_batch:  43    -0.0106794    0.275434
total_batch:  44    -0.0115777    0.299923
total_batch:  45    -0.0107367    0.276901
total_batch:  45 test_loss:  7.26396
Groud-Truth: 4.06752
total_batch:  46    -0.0117619    0.315244
total_batch:  47    -0.0125314    0.32044
total_batch:  48    -0.0128803    0.31028
total_batch:  49    -0.0124846    0.319674
total_batch:  50    -0.0125977    0.318701
total_batch:  50 test_loss:  7.27055
Groud-Truth: 4.07531
total_batch:  51    -0.012405    0.318317
total_batch:  52    -0.0119562    0.300431
total_batch:  53    -0.0122883    0.314295
total_batch:  54    -0.0126281    0.319951
total_batch:  55    -0.0127594    0.316178
total_batch:  55 test_loss:  7.22564
Groud-Truth: 4.07341
total_batch:  56    -0.0124642    0.313935
total_batch:  57    -0.012497    0.297185
total_batch:  58    -0.0123365    0.302482
total_batch:  59    -0.0123436    0.328591
total_batch:  60    -0.0124708    0.318689
total_batch:  60 test_loss:  7.2112
Groud-Truth: 4.07617
total_batch:  61    -0.0122884    0.284387
total_batch:  62    -0.0121964    0.305566
total_batch:  63    -0.011934    0.291181
total_batch:  64    -0.0120234    0.309875
total_batch:  65    -0.0125715    0.308552
total_batch:  65 test_loss:  7.28753
Groud-Truth: 4.07621
total_batch:  66    -0.0128219    0.31488
total_batch:  67    -0.0126224    0.307146
total_batch:  68    -0.0131105    0.32626
total_batch:  69    -0.0121081    0.311348
total_batch:  70    -0.0122315    0.310016
total_batch:  70 test_loss:  7.29423
Groud-Truth: 4.07411
total_batch:  71    -0.0130023    0.331268
total_batch:  72    -0.012701    0.320357
total_batch:  73    -0.0125991    0.324347
total_batch:  74    -0.0121278    0.306766
total_batch:  75    -0.0121153    0.311375
total_batch:  75 test_loss:  7.28657
Groud-Truth: 4.07812
total_batch:  76    -0.0130154    0.314399
total_batch:  77    -0.0129696    0.331336
total_batch:  78    -0.0128378    0.334735
total_batch:  79    -0.0129301    0.329491
total_batch:  80    -0.0122206    0.307798
total_batch:  80 test_loss:  7.39785
Groud-Truth: 4.06664
total_batch:  81    -0.0124936    0.307753
total_batch:  82    -0.0128268    0.331966
total_batch:  83    -0.0133842    0.343977
total_batch:  84    -0.0133574    0.340249
total_batch:  85    -0.0130636    0.329189
total_batch:  85 test_loss:  7.36811
Groud-Truth: 4.07863
total_batch:  86    -0.013543    0.340774
total_batch:  87    -0.013276    0.335258
total_batch:  88    -0.0134741    0.341645
total_batch:  89    -0.0132732    0.332504

total_batch:  90    -0.0132484    0.343856
total_batch:  90 test_loss:  7.3697
Groud-Truth: 4.0741
total_batch:  91    -0.0136935    0.342951
total_batch:  92    -0.0139279    0.33818
total_batch:  93    -0.0136782    0.345494
total_batch:  94    -0.0138809    0.350263
total_batch:  95    -0.013651    0.350747
total_batch:  95 test_loss:  7.35818


#### Length 20:
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


