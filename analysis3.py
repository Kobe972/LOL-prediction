from pandas import Series,DataFrame
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
main_data=pd.read_csv('data.csv')
subtable=main_data['player1_championId'].value_counts()+main_data['player2_championId'].value_counts()+main_data['player3_championId'].value_counts()+main_data['player4_championId'].value_counts()+main_data['player5_championId'].value_counts()+main_data['player6_championId'].value_counts()+main_data['player7_championId'].value_counts()+main_data['player8_championId'].value_counts()+main_data['player9_championId'].value_counts()+main_data['player10_championId'].value_counts()
subtable=subtable.sort_index(ascending=True)
subtable=subtable[:45]
fig, ax = plt.subplots()
plt.title('player_championId')
subtable.plot(kind='bar')
ax.set_xticks(np.arange(min(subtable.index),max(subtable.index),5))
plt.xlim(min(subtable.index),max(subtable.index))
plt.xlabel('ID')
plt.ylabel('count')
plt.show()



