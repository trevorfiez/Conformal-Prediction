# Conformal Prediction for CIFAR-10 Using a Log Loss for the Final Layer #

I trained the network for 100,000 batches using SGD with just the default parameters from the example.

## Model precision and recall for a model trained with class weights ##

For the entire network where I would check to see that the label was correctly labeled 1 but could also output multiple
postive labels:

Recall = 0.946

Precision = 0.578

Just using the top-k highest scoring labels though the precision and recall are as follows:

| Top k | Recall | Precision |
| :---: | :---: | :---: |
| 1 | 0.812 | 0.812 |
| 2 | 0.927 | 0.463 |
| 3 | 0.966 | 0.322 |
| 4 | 0.982 | 0.246 |
| 5 | 0.991 | 0.198 |
| 6 | 0.995 | 0.166 |
| 7 | 0.997 | 0.143 |
| 8 | 0.999 | 0.125 |
| 9 | 0.999 | 0.111 |

## Model trained without sample weights ##

The recall and precision from the output of the network is:

Recall = 0.796

Precision = 0.895.

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


## Tables for Different Weights ##

Each table element is formatted as recall / precision / average set size.

The title for each table denotes the weight on negative samples.

### Weight 1.o ###

| k | 1 | 2 | 3 | 4 | 5 | 6 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: 
| Top-k | 0.851 / 0.851 / 1.00 | 0.938 / 0.469 / 2.00 | 0.972 / 0.324 / 3.00 | 0.985 / 0.246 / 4.00 | 0.992 / 0.198 / 5.00 | 0.996 / 0.166 / 6.00 |
| All-threshold | 0.847 / 0.858 / 0.99 | 0.933 / 0.680 / 1.37 | 0.972 / 0.520 / 1.87 | 0.986 / 0.420 / 2.35 | 0.993 / 0.348 / 2.85 | 0.996 / 0.290 / 3.43 |
| Class Threshold | 0.847 / 0.858 / 0.99 | 0.933 / 0.680 / 1.37 | 0.972 / 0.520 / 1.87 | 0.986 / 0.420 / 2.35 | 0.993 / 0.348 / 2.85 | 0.996 / 0.290 / 3.43 |
| Diff Pre-sigmoid | 0.849 / 0.861 / 0.99 | 0.935 / 0.677 / 1.38 | 0.972 / 0.493 / 1.97 | 0.983 / 0.393 / 2.51 | 0.991 / 0.322 / 3.08 | 0.996 / 0.244 / 4.08 |
| Diff | 0.847 / 0.856 / 0.99 | 0.934 / 0.537 / 1.74 | 0.970 / 0.326 / 2.98 | 0.982 / 0.264 / 3.72 | 0.990 / 0.224 / 4.43 | 0.995 / 0.190 / 5.23 |
| Ratio | 0.846 / 0.855 / 0.99 | 0.933 / 0.680 / 1.37 | 0.970 / 0.519 / 1.87 | 0.986 / 0.417 / 2.37 | 0.992 / 0.354 / 2.80 | 0.997 / 0.284 / 3.51 |
