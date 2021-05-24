import numpy as np #生成图上所需数据点，生成正弦曲线
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt #画图
from pandas import Series,DataFrame
import pandas as pd
from IPython.display import Image  
from sklearn import tree
import pydotplus

DATASET_NAME='data.csv'
LABEL_NAME='label.csv'

def load_data(begin_row,end_row):
    data=pd.read_csv(DATASET_NAME)
    origin=data.iloc[begin_row:end_row]
    _boolean=origin.iloc[:,5:11]
    ret=np.array([_boolean.iloc[i].values for i in range(end_row-begin_row)])
    numbers={}
    numbers['player_kills']=(origin['player1_kills']+origin['player2_kills']+origin['player3_kills']+origin['player4_kills']+origin['player5_kills']).values
    numbers['player_deaths']=(origin['player1_deaths']+origin['player2_deaths']+origin['player3_deaths']+origin['player4_deaths']+origin['player5_deaths']).values
    numbers['player_assists']=(origin['player1_assists']+origin['player2_assists']+origin['player3_assists']+origin['player4_assists']+origin['player5_assists']).values
    numbers['player_goldEarned']=(origin['player1_goldEarned']+origin['player2_goldEarned']+origin['player3_goldEarned']+origin['player4_goldEarned']+origin['player5_goldEarned']).values
    for key,value in numbers.items():
        numbers[key]=(value-value.min())/(value.max()-value.min())
        ret=np.concatenate((ret,np.array([numbers[key]]).T),axis=1)
    return ret

def load_label():
    data=pd.read_csv(LABEL_NAME)
    labels=data['gameDuration'].values
    return labels

'''
#used for testing
training_data=load_data(20000,68000)
labels=load_label()
regr = DecisionTreeRegressor(max_depth=9)
regr=regr.fit(training_data,labels[:48000])

test_data=load_data(68000,80000)
pre1=regr.predict(test_data)
print('mean-square on test data:',np.sum(np.square(pre1-labels[48000:]))/len(pre1))

pre2=regr.predict(training_data)
print('mean-square on training data:',np.sum(np.square(pre2-labels[:48000]))/len(pre2))
'''
training_data=load_data(20000,80000)
labels=load_label()
regr = DecisionTreeRegressor(max_depth=9)
regr=regr.fit(training_data,labels)

pre2=regr.predict(training_data)
print('mean-square on training data:',np.sum(np.square(pre2-labels))/len(pre2))

predict_data=load_data(0,20000)
pre=regr.predict(predict_data)
pd_data = pd.DataFrame(pre,columns=['gameDuration'])
pd_data.index=list(range(0,20000))
pd_data.to_csv('prediction_regtree.csv')

dot_data = tree.export_graphviz(regr, out_file=None,  #regr是对应分类器
                         feature_names=['team1_firstBlood','team1_firstTower','team1_firstInhibitor','team1_firstBaron','team1_firstDragon','team1_firstRiftHerald','player_kills','player_deaths','player_assists','player_goldEarned'],   #对应特征的名字
                         class_names=['team1_win'],    #对应类别的名字
                         filled=True, rounded=True,  
                         special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data)  
graph.write_pdf('result.pdf') 
