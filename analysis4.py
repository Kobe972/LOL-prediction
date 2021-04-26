from pandas import Series,DataFrame
import pandas as pd
import math
import matplotlib.pyplot as plt
main_data=pd.read_csv('data.csv')
def win_to_team_data(data_name1,data_name2):
    class1=main_data[data_name2][main_data[data_name1]]
    class2=main_data[data_name2][-main_data[data_name1]]
    result=DataFrame([class1.value_counts(),class2.value_counts()],index=[data_name1,'!'+data_name1])
    result.columns.name=data_name2
    result['sum']=[result.loc[data_name1].sum(),result.loc['!'+data_name1].sum()]
    print(result)
    p1=float(result[0][data_name1])/result['sum'][data_name1]
    p2=float(result[0]['!'+data_name1])/result['sum']['!'+data_name1]
    entropy1=-p1*math.log2(p1)-(1-p1)*math.log2(1-p1)
    entropy2=-p2*math.log2(p2)-(1-p2)*math.log2(1-p2)
    print('weighted entropy=',entropy1*result['sum'][data_name1]/result['sum'].sum()+entropy2*result['sum']['!'+data_name1]/result['sum'].sum())
    print('weighted accuracy=',(1-p1)*result['sum'][data_name1]/result['sum'].sum()+p2*result['sum']['!'+data_name1]/result['sum'].sum())
    result2=result.drop(columns=['sum'])
    result2.plot(kind='barh',)
    plt.savefig('results\\histogram.png',bbox_inches='tight')
    plt.show()

names=['team1_firstBlood','team1_firstTower','team1_firstInhibitor','team1_firstBaron','team1_firstDragon','team1_firstRiftHerald']
for i in range(0,len(names)-1):
    for j in range(i+1,len(names)):
        win_to_team_data(names[i],names[j])
