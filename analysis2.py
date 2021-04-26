from pandas import Series,DataFrame
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
def player_data_classifier(attr,reverse=0):
    main_data=pd.read_csv('data.csv')
    subtable_kill=main_data['player1_'+attr]+main_data['player2_'+attr]+main_data['player3_'+attr]+main_data['player4_'+attr]+main_data['player5_'+attr]
    subtable_kill.name='player_'+attr
    subtable=DataFrame([main_data['team1_win'],subtable_kill])
    subtable=subtable.stack().unstack(0)
    subtable.index.name='index'
    m=min(subtable_kill)
    M=max(subtable_kill)
    max_accuracy=0
    max_i=m
    def no(x):
        if(x==0):
            return 1
        else:
            return 0
    for i in range(m,M+1):
        if(M-m>=40):
            if((i-m)%100==0):
                print('This is the ',i-m,'/',M-m,'th iteration.')
        cross_entropy=0
        tmp1=subtable['player_'+attr]>=i
        tmp2=subtable['player_'+attr]<i
        u=subtable['team1_win'][tmp1]
        v=subtable['team1_win'][tmp2]
        if(reverse==0):
            accuracy=(u.sum()+v.apply(no).sum())/len(subtable.index)
        else:
            accuracy=(v.sum()+u.apply(no).sum())/len(subtable.index)
        if accuracy>max_accuracy:
            max_accuracy=accuracy
            max_i=i
    print('max_accuracy=',max_accuracy)
    print('corresponding classifier i=',max_i)
    tmp=subtable[subtable['player_'+attr]>=max_i]['team1_win'].value_counts()
    if(reverse==0):
        tmp.index=['team1_win','team1_loss']
    else:
        tmp.index=['team1_loss','team1_win']
    plt.title('player_'+attr+'>='+str(max_i))
    tmp.plot(kind='barh')
    plt.show()
    print('player_'+attr+'>=',max_i,'对应的玩家胜负情况统计')
    print(tmp)
    tmp=subtable[subtable['player_'+attr]<max_i]['team1_win'].value_counts()
    if(reverse==0):
        tmp.index=['team1_loss','team1_win']
    else:
        tmp.index=['team1_win','team1_loss']
    plt.title('player_'+attr+'<'+str(max_i))
    tmp.plot(kind='barh')
    plt.show()
    print('player_'+attr+'<',max_i,'对应的玩家胜负情况统计')
    print(tmp)
player_data_classifier('kills',0)


