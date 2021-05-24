import numpy as np
from sklearn.decomposition import PCA
from pandas import Series,DataFrame
import pandas as pd
import tensorflow as tf

DATASET_NAME='data.csv'
LABEL_NAME='label.csv'
pca=PCA(n_components=2)

def load_data(begin_row,end_row): #读取数据
    data=pd.read_csv(DATASET_NAME)
    origin=data.iloc[begin_row:end_row]
    _boolean=origin.iloc[:,5:11]
    _ret=np.array([_boolean.iloc[i].values for i in range(end_row-begin_row)])
    ret=pca.fit_transform(_ret) #将前面几个布尔类型的变量做主成分分析，压缩成2个
    numbers={}
    numbers['player_kills']=(origin['player1_kills']+origin['player2_kills']+origin['player3_kills']+origin['player4_kills']+origin['player5_kills']).values
    numbers['player_deaths']=(origin['player1_deaths']+origin['player2_deaths']+origin['player3_deaths']+origin['player4_deaths']+origin['player5_deaths']).values
    numbers['player_assists']=(origin['player1_assists']+origin['player2_assists']+origin['player3_assists']+origin['player4_assists']+origin['player5_assists']).values
    numbers['player_goldEarned']=(origin['player1_goldEarned']+origin['player2_goldEarned']+origin['player3_goldEarned']+origin['player4_goldEarned']+origin['player5_goldEarned']).values
    for key,value in numbers.items():
        numbers[key]=(value-value.min())/(value.max()-value.min())
        ret=np.concatenate((ret,np.array([numbers[key]]).T),axis=1)
    return ret

def load_label(): #读取标签
    data=pd.read_csv(LABEL_NAME)
    labels=data['gameDuration'].values
    return np.array([labels]).T

def train(x,labels): #训练
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(0.1).minimize(loss)
    # 初始化所有变量
    init = tf.global_variables_initializer()
    sess.run(init)
    # 迭代次数 = 30000
    for i in range(30000):
        # 训练
        sess.run(train_step, feed_dict={xs: x, ys: labels})
        if i % 100 == 0:
            print('epoch;',i/100,'loss:',sess.run(loss, feed_dict={xs: x, ys: labels}))

def compute_accuracy(x,labels): #计算mean-square
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    print('mean-square error on testing set:',sess.run(loss, feed_dict={xs: x, ys: labels}))

def predict(x): #对输入数据x预测
    result=sess.run(prediction,feed_dict={xs:x})
    ##写入文件
    pd_data = pd.DataFrame(result,columns=['gameDuration'])
    pd_data.index=list(range(0,20000))
    pd_data.to_csv('prediction.csv')

xs = tf.placeholder(tf.float32, [None, 6])
ys = tf.placeholder(tf.float32, [None, 1])
# 隐层1
Weights1 = tf.Variable(tf.random_normal([6, 5]))
biases1 = tf.Variable(tf.zeros([1, 5]) + 0.1)
Wx_plus_b1 = tf.matmul(xs, Weights1) + biases1
l1 = tf.nn.sigmoid(Wx_plus_b1)
# 隐层2
Weights2 = tf.Variable(tf.random_normal([5, 15]))
biases2 = tf.Variable(tf.zeros([1, 15]) + 0.1)
Wx_plus_b2 = tf.matmul(l1, Weights2) + biases2
l2 = tf.nn.sigmoid(Wx_plus_b2)
# 输出层
Weights3 = tf.Variable(tf.random_normal([15, 1]))
biases3 = tf.Variable(tf.zeros([1, 1]) + 0.1)
prediction = tf.matmul(l2, Weights3) + biases3
sess=tf.Session()
'''
#used for testing
training_data=load_data(20000,68000)
labels=load_label()
train(training_data,labels[:48000,:])

test_data=load_data(68000,80000)
compute_accuracy(test_data,labels[48000:,:])

predict_data=load_data(0,20000)
predict(predict_data)
'''
training_data=load_data(20000,80000)
labels=load_label()
train(training_data,labels)

predict_data=load_data(0,20000)
predict(predict_data)

