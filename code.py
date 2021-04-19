from pandas import Series,DataFrame
import pandas as pd
import math
main_data=pd.read_csv('data.csv')
def win_to_team_data(data_name):
    class1=main_data['team1_win'][main_data[data_name]]
    class2=main_data['team1_win'][-main_data[data_name]]
    result=DataFrame([class1.value_counts(),class2.value_counts()],index=[data_name,'!'+data_name])
    result.columns.name='team1_win'
    result['sum']=[result.loc[data_name].sum(),result.loc['!'+data_name].sum()]
    print(result)
    p1=float(result[0][data_name])/result['sum'][data_name]
    p2=float(result[0]['!'+data_name])/result['sum']['!'+data_name]
    entropy1=-p1*math.log2(p1)-(1-p1)*math.log2(1-p1)
    entropy2=-p1*math.log2(p2)-(1-p1)*math.log2(1-p2)
    print('mean entropy=',entropy1*result['sum'][data_name]/result['sum'].sum()+entropy2*result['sum']['!'+data_name]/result['sum'].sum())
    print('accuracy=',(1-p1)*result['sum'][data_name]/result['sum'].sum()+p2*result['sum']['!'+data_name]/result['sum'].sum())

win_to_team_data('team1_firstBlood')
