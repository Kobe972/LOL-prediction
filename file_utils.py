from pandas import Series,DataFrame
import pandas as pd
import numpy as np
DATASET_NAME='data.csv'
def read_data(begin_row,end_row):
    properties={}
    origin=pd.read_csv(DATASET_NAME)
    main_data=origin.iloc[begin_row:end_row,4:]
    properties['class']=main_data[main_data.columns[0]].values
    for name in main_data.columns[1:7]:
        properties[name]=main_data[name].values
    properties['player_kills']=(main_data['player1_kills']+main_data['player2_kills']+main_data['player3_kills']+main_data['player4_kills']+main_data['player5_kills']).values
    properties['player_deaths']=(main_data['player1_deaths']+main_data['player2_deaths']+main_data['player3_deaths']+main_data['player4_deaths']+main_data['player5_deaths']).values
    properties['player_assists']=(main_data['player1_assists']+main_data['player2_assists']+main_data['player3_assists']+main_data['player4_assists']+main_data['player5_assists']).values
    properties['player_goldEarned']=(main_data['player1_goldEarned']+main_data['player2_goldEarned']+main_data['player3_goldEarned']+main_data['player4_goldEarned']+main_data['player5_goldEarned']).values
    types=np.array([0,0,0,0,0,0,1,1,1,1])
    return types,properties
