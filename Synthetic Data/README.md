# LeakGAN

## Requirements: 
* **Tensorflow r1.2.1**
* Python 2.7
* CUDA 7.5+ (For GPU)

## Introduction
Apply Generative Adversarial Nets and Hierarchical Reinforcement Learning to generating sequences of discrete tokens using leaked information from D.

## File

LeakGAN40.py : The main file for length 40.
LeakGAN20.py : The main file for length 20.

Model.py: Our model.

target_lstm20.py : The oracle model for length 20.
target_lstm40.py : The oracle model for length 40.

data_loader20.py: Data helpy function for length 20.
data_loader40.py: Data helpy function for length 40.

## Details 
We provide example codes to repeat the synthetic data experiments with oracle evaluation mechanisms in length 20 and 40.
To run the experiment with default parameters for length 20:
```
$ python LeakGAN.py
```

To run the experiment with default parameters for length 20:
```
$ python LeakGAN40.py
```

In our code, we
The experiment has two stages. In the first stage, use the positive data provided by the oracle and Maximum Likelihood Estimation to perform supervise learning. In the second stage, use adversarial training to improve the generator.

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
pre-train epoch  0 test_loss  9.79046
pre-train epoch  5 test_loss  9.23233
pre-train epoch  10 test_loss  9.01165
pre-train epoch  15 test_loss  8.55405
pre-train epoch  20 test_loss  8.37889
pre-train epoch  25 test_loss  8.25284
pre-train epoch  30 test_loss  8.26965
pre-train epoch  35 test_loss  8.2791
pre-train epoch  40 test_loss  8.25489
pre-train epoch  45 test_loss  8.24959
pre-train epoch  50 test_loss  8.2421
pre-train epoch  55 test_loss  8.27822
pre-train epoch  60 test_loss  8.29729
pre-train epoch  65 test_loss  8.33883
pre-train epoch  70 test_loss  8.35861
pre-train epoch  75 test_loss  8.37023
pre-train epoch  80 test_loss  8.36275
pre-train epoch  85 test_loss  8.35861
pre-train epoch  90 test_loss  8.34861
pre-train epoch  95 test_loss  8.35861
pre-train epoch  100 test_loss  8.35861
pre-train epoch  105 test_loss  8.33861
pre-train epoch  110 test_loss  8.34861
pre-train epoch  115 test_loss  8.35861
#########################################################################
total_batch:  0 test_loss:  8.32261
target: -0.00115139
Groud-Truth: 5.76903
0
total_batch:  1    -0.0748189    2.65353
1
total_batch:  2    -0.0733639    2.59667
1
total_batch:  3    -0.0750953    2.63402
2
total_batch:  4    -0.072135    2.55991
2
total_batch:  5    -0.0711216    2.5347
total_batch:  5 test_loss:  8.06178
Groud-Truth: 5.74315
3
total_batch:  6    -0.0694847    2.51656
3
total_batch:  7    -0.0692976    2.41641
4
total_batch:  8    -0.0708733    2.49223
4
total_batch:  9    -0.0702165    2.36559
5
total_batch:  10    -0.0672321    2.32977
total_batch:  10 test_loss:  7.91655
Groud-Truth: 5.75286
5
total_batch:  11    -0.0681583    2.41653
6
total_batch:  12    -0.0654267    2.34092
6
total_batch:  13    -0.0687298    2.33515
7
total_batch:  14    -0.0682873    2.25028
7
total_batch:  15    -0.0663429    2.22504
total_batch:  15 test_loss:  7.79723
Groud-Truth: 5.75233
8
total_batch:  16    -0.0650572    2.21478
8
total_batch:  17    -0.0671644    2.16018
9
total_batch:  18    -0.0652224    2.1117
9
total_batch:  19    -0.0641908    2.02485
10
total_batch:  20    -0.0614942    1.92095
total_batch:  20 test_loss:  7.44647
Groud-Truth: 5.75483
10
total_batch:  21    -0.0599244    1.92014
11
total_batch:  22    -0.0557595    1.82898
11
total_batch:  23    -0.0582773    1.78447
12
total_batch:  24    -0.0548887    1.70569
12
total_batch:  25    -0.0596043    1.91641
total_batch:  25 test_loss:  7.37603
Groud-Truth: 5.74729
13
total_batch:  26    -0.0565431    1.84316
13
total_batch:  27    -0.0580525    1.80003
14
total_batch:  28    -0.0555371    1.73949
14
total_batch:  29    -0.0560679    1.84264
15
total_batch:  30    -0.0570525    1.71439
en: -0.00112237
total_batch:  30 test_loss:  7.23705
target: -0.00115106
Groud-Truth: 5.75674
15
total_batch:  31    -0.059762    1.78002
16
total_batch:  32    -0.0528225    1.68505
16
total_batch:  33    -0.053743    1.72411
17
total_batch:  34    -0.0586561    1.73525
17
total_batch:  35    -0.053348    1.73746
total_batch:  35 test_loss:  7.17774
Groud-Truth: 5.75209
18
total_batch:  36    -0.0545638    1.62786
18
total_batch:  37    -0.0557342    1.6338
19
total_batch:  38    -0.0539989    1.61291
19
total_batch:  39    -0.0557821    1.57712
20
total_batch:  40    -0.0531582    1.62076
total_batch:  40 test_loss:  7.08687
Groud-Truth: 5.7461
20
total_batch:  41    -0.0558011    1.64124
21
total_batch:  42    -0.0518572    1.56386
21
total_batch:  43    -0.051937    1.55573
22
total_batch:  44    -0.0560074    1.6642
22
total_batch:  45    -0.0480089    1.55876
total_batch:  45 test_loss:  7.10859
Groud-Truth: 5.75255
23
total_batch:  46    -0.0496448    1.56284
23
total_batch:  47    -0.0507183    1.61702
24
total_batch:  48    -0.0493574    1.57254
24
total_batch:  49    -0.0485481    1.55289
25
total_batch:  50    -0.0494223    1.49561
total_batch:  50 test_loss:  7.11554
Groud-Truth: 5.76044
25
total_batch:  51    -0.0501754    1.57039
26
total_batch:  52    -0.0482378    1.55509
26
total_batch:  53    -0.0476559    1.60568
27
total_batch:  54    -0.052231    1.49869
27
total_batch:  55    -0.0460759    1.50756
en: -0.00112776
total_batch:  55 test_loss:  7.0391
target: -0.00115031
Groud-Truth: 5.73975
28
total_batch:  56    -0.0489183    1.41079
28
total_batch:  57    -0.046071    1.39837
29
total_batch:  58    -0.0470096    1.45003
29
total_batch:  59    -0.0483027    1.60029
30
total_batch:  60    -0.0484383    1.56495
total_batch:  60 test_loss:  7.08765
Groud-Truth: 5.75295
30
total_batch:  61    -0.0511004    1.55408
31
total_batch:  62    -0.0459844    1.51821
31
total_batch:  63    -0.0430586    1.43311
32
total_batch:  64    -0.0477675    1.53567
32
total_batch:  65    -0.0446857    1.47013
en: -0.00113196
total_batch:  65 test_loss:  7.13849
target: -0.00115119
Groud-Truth: 5.75425
33
total_batch:  66    -0.0511859    1.58795
33
total_batch:  67    -0.0499813    1.58719
34
total_batch:  68    -0.0470622    1.54053
34
total_batch:  69    -0.0456274    1.50734
35
total_batch:  70    -0.0460732    1.41583
total_batch:  70 test_loss:  6.97844
Groud-Truth: 5.75092
35
total_batch:  71    -0.0465902    1.50938
36
total_batch:  72    -0.0411174    1.50827
36
total_batch:  73    -0.0461697    1.41665
37
total_batch:  74    -0.0443668    1.41308
37
total_batch:  75    -0.0454291    1.6019
total_batch:  75 test_loss:  7.07213
Groud-Truth: 5.75355
38
total_batch:  76    -0.0387431    1.56111
38
total_batch:  77    -0.041546    1.44214
39
total_batch:  78    -0.0418215    1.47762
39
total_batch:  79    -0.0378464    1.41147
40
total_batch:  80    -0.0417305    1.51489
total_batch:  80 test_loss:  6.94508
Groud-Truth: 5.75176
40
total_batch:  81    -0.0456369    1.49778
41
total_batch:  82    -0.0346479    1.45052
41
total_batch:  83    -0.033707    1.47873
42
total_batch:  84    -0.039991    1.45967
42
total_batch:  85    -0.0361866    1.45741
en: -0.00114029
total_batch:  85 test_loss:  7.24787
target: -0.00115056
Groud-Truth: 5.75004
43
total_batch:  86    -0.0416327    1.44945
43
total_batch:  87    -0.0391876    1.46167
44
total_batch:  88    -0.0341999    1.52573
44
total_batch:  89    -0.0383887    1.54944
45
total_batch:  90    -0.0362922    1.44933
total_batch:  90 test_loss:  7.1182
Groud-Truth: 5.74758
45
total_batch:  91    -0.0361173    1.48233
46
total_batch:  92    -0.0345551    1.57137
46
total_batch:  93    -0.0420464    1.56493
47
total_batch:  94    -0.039561    1.54575
47
total_batch:  95    -0.0356424    1.61081
total_batch:  95 test_loss:  7.36414
Groud-Truth: 5.74691
48
total_batch:  96    -0.0273507    1.54172
48
total_batch:  97    -0.0357092    1.5455
49
total_batch:  98    -0.033188    1.56247
49
total_batch:  99    -0.0375338    1.59204
50
total_batch:  100    -0.0322227    1.58326
total_batch:  100 test_loss:  7.22594
Groud-Truth: 5.75549
50
total_batch:  101    -0.0329372    1.42671
51
total_batch:  102    -0.0322859    1.48235


