# LOL-prediction
## Introduction
This is an assignment for Data Analysis,aiming to predict the result of LOL using the data of each team.
### Directory
results:Contains generated graphs,results after data processing,etc.  
analysisN.py:They are independent.Each file analyses the data from a unique perspective.  
Below is the second task--write a program to predict the result.
decision_tree.py:Written module for decision tree
file_utils:used to load csv files and return data structure corresponding to decision_tree.py  
predict.py:Verify the accuracy of the module.If not pruned,the accuracy will be 94.725% according to the division of 4:1(train_data:test_data).If pruned,95.7775%.
