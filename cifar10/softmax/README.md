## Conformal Prediction for CIFAR-10 using a softmax classification layer ##

I was kind of confused on the measures to actually run conformal prediction on the cifar-10 dataset so I just
am reporting top-k recall and precision. Maybe we can go over how to actually do the conformal prediction stuff sometime.

I trained the network for 100,000 batches using SGD with just the default parameters from the example.


| Top k | Recall | Precision |
| :---: | :---: | :---: |
| 1 | 0.86 | 0.86 |
| 2 | 0.946 | 0.473 |
| 3 | 0.974 | 0.325 |
| 4 | 0.988 | 0.247 |
| 5 | 0.993 | 0.199 |
| 6 | 0.997 | 0.166 |
| 7 | 0.998 | 0.143 |
| 8 | 0.999 | 0.125 |
| 9 | 1.000 | 0.111 |
