## Conformal Prediction for CIFAR-10 Using a Log Loss for the Final Layer ##

I trained the network for 100,000 batches using SGD with just the default parameters from the example.

The recall and precision from the output of the network is:
Recall = 0.796, Precision = 0.895.

The top-k precision and recall is as follows:


| Top k | Recall | Precision |
| :---: | :---: | :---: |
| 1 | 0.848 | 0.86 |
| 2 | 0.941 | 0.473 |
| 3 | 0.970 | 0.325 |
| 4 | 0.985 | 0.247 |
| 5 | 0.993 | 0.199 |
| 6 | 0.996 | 0.166 |
| 7 | 0.998 | 0.143 |
| 8 | 0.999 | 0.125 |
| 9 | 1.000 | 0.111 |
