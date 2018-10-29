# CSCI5525_hw2
### Packages used
* os, sys, time, math
* argparse
* numpy
* matplotlib.pyplot
### Datasets
* MNIST-13.csv: 2000 x 784, 2 classes (1 & 3)
### Run the command for problem 1:
The function myDualSVM(filename, C) has 2 inputs as described in assignment.<br>
The list of C ([0.01, 0.1, 1, 10, 100]) is hard coded in main().
```
python3 myDualSVM.py MNIST-13.csv

```
The myDualSVM.py outputs result to myDualSVM_result.csv.

### Run the command for problem 2:
THe function myPegasos(filename, k, numruns) and mySoftplus(filename, k, numruns) both have 3 inputs as described in assignment.<br>
The list of k ([1, 20, 200, 1000, 2000]) is hard coded in main().
```
python3 myPegasos.py MNIST-13.csv 5
```
The myPegasos.py outputs result to myPegasos_result.csv. And save plots in img/<k>batch_myPegasos.png
  
```
python3 mySoftplus.py MNIST-13.csv 5
```
The mySoftplus.py outputs result to mySoftplus_result.csv. And save plots in img/<k>batch_mySoftplus.png
