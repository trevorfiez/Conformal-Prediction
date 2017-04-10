
# Results for Softmax with Cifar-100 #


| k | 1 | 2 | 3 | 4 | 5 | 6 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: 
| Top-k | 0.577 / 0.577 / 1.00 | 0.706 / 0.353 / 2.00 | 0.773 / 0.258 / 3.00 | 0.812 / 0.203 / 4.00 | 0.842 / 0.168 / 5.00 | 0.862 / 0.144 / 6.00 |
| All-threshold | 0.575 / 0.623 / 0.92 | 0.700 / 0.432 / 1.62 | 0.773 / 0.325 / 2.38 | 0.815 / 0.263 / 3.10 | 0.843 / 0.221 / 3.81 | 0.863 / 0.194 / 4.45 |
| Class Threshold | 0.575 / 0.623 / 0.92 | 0.700 / 0.432 / 1.62 | 0.773 / 0.325 / 2.38 | 0.815 / 0.263 / 3.10 | 0.843 / 0.221 / 3.81 | 0.863 / 0.194 / 4.45 |
| Diff Pre-sigmoid | 0.570 / 0.583 / 0.98 | 0.705 / 0.384 / 1.83 | 0.770 / 0.289 / 2.66 | 0.811 / 0.235 / 3.46 | 0.846 / 0.196 / 4.32 | 0.864 / 0.170 / 5.08 |
| Diff | 0.571 / 0.584 / 0.98 | 0.702 / 0.113 / 6.20 | 0.768 / 0.056 / 13.73 | 0.809 / 0.041 / 19.62 | 0.837 / 0.033 / 25.16 | 0.859 / 0.029 / 29.79 |
| Ratio | 0.569 / 0.583 / 0.98 | 0.704 / 0.385 / 1.83 | 0.770 / 0.289 / 2.66 | 0.811 / 0.235 / 3.46 | 0.846 / 0.196 / 4.32 | 0.864 / 0.170 / 5.08 |