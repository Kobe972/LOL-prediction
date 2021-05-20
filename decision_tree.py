import numpy as np
from pandas import Series,DataFrame
import pickle
import math
MAX_DEPTH=10
class ClsProperty:
    """
    self.name:类型名，用于索引
    self.type:分类属性类型，0表示离散，1表示连
    self.value:属性的值
    self.thresh:连续属性分类阈值
    """
    def __init__(self,name,ctype,values=np.empty(0)):
        self.name=name
        self.type=ctype #类型为int32
        self.values=values

        
class Data:
    def __init__(self,ctype,properties={}):
        self.properties=properties
        self.type=ctype #类型为ndarray
        self.count=len(properties['class']) #共有多少组数据
        assert len(self.type)==len(properties.keys())-1
        
    def add_property(self,info):
        for key,values in info:
            self.properties[key]=self.properties[key].append(info)
            
class Node:
    """
    self.data:节点存放的数据
    self.clsproperties:分类属性
    self.classifier:分类标准
    self.clsproperties:分类依据
    self.ancestor:公共祖先
    """
    def __init__(self,classes,data,depth,ancestor=None,clsproperties=None): #classes表示会分成哪几类，data表示该节点包含的数据
        self.classes=classes
        self.num_cls=len(classes)
        self.data=data
        self.child=[]
        self.leaf=0
        self.depth=depth
        self.clsproperties=clsproperties
        if clsproperties==None:
            self.clsproperties=[]
            self.gen_clsproperties()
        if ancestor==None:
            self.ancestor=self
        else:
            self.ancestor=ancestor
    def gen_clsproperties(self):
        index=0
        for key,value in self.data.properties.items():
            if key=='class':
                continue
            if self.data.type[index]==0:
                self.clsproperties.append(ClsProperty(key,0,np.unique(value)))
            else:
                self.clsproperties.append(ClsProperty(key,0))
            index=index+1
            
    def entropy(self,data):#计算熵
        p=np.zeros(self.num_cls)
        for i in range(self.num_cls):
            p[i]=np.sum(data.properties['class']==self.classes[i])/data.count
            if(p[i]!=0):
                p[i]=p[i]*math.log2(p[i])
            p[i]=-p[i]
        return p.sum()
    
    def ent_gain(self,classifier):#计算信息增益
        origin=self.entropy(self.data)
        new=0
        if classifier.type==0:#离散型变量
            for value in classifier.values:
                mask=(self.data.properties[classifier.name]==value)
                properties={}
                for key in self.data.properties.keys():
                    properties[key]=self.data.properties[key][mask]
                tmp=Data(self.data.type,properties)
                new+=tmp.count/self.data.count*self.entropy(tmp)
        else:#连续型变量
            mask=(self.data.properties[classifier.name]>=classifier.threshold)
            properties={}
            for key in self.data.properties.keys():
                properties[key]=self.data.properties[key][mask]
            tmp=Data(properties)
            new+=tmp.count/self.data.count*self.entropy(tmp)
            mask=(self.data.properties[classifier.name]<classifier.threshold)
            properties={}
            for key in self.data.properties.keys():
                properties[key]=self.data.properties[key][mask]
            tmp=Data(self.data.type,properties)
            new+=tmp.count/self.data.count*self.entropy(tmp)
        return origin-new
    
    def most_class(self):
        ans=[]
        for cls in self.classes:
            mask=self.data.properties['class']==cls
            ans.append(np.sum(mask))
        return self.classes[np.argmax(ans)]
    
    def predict(self,prop): #根据属性prop预测结果
        if self.leaf==1:
            return self.result
        if self.classifier.type==0:
            for i in range(len(self.classifier.values)):
                if prop[self.classifier.name]==self.classifier.values[i]:
                    return self.child[i].predict(prop)
        else:
            if prop[self.classifier.name]>=self.classifier.thresh:
                return self.child[0].predict(prop)
            else:
                return self.child[1].predict(prop)
    def compute_accuracy(self,dataset): #在dataset数据集（类型为Data）上计算分类的精确度
        accuracy=0
        for i in range(dataset.count):
            prop={}
            for key,value in dataset.properties.items():
                prop[key]=value[i]
            prediction=self.predict(prop)
            if prediction==dataset.properties['class'][i]:
                accuracy+=1
        accuracy=accuracy/dataset.count
        return accuracy
    
    def IV(self,clsprop):
        summary=0
        for value in clsprop.values:
            mask=(self.data.properties[clsprop.name]==value)
            tmp=mask.sum()
            tmp=tmp/self.data.count
            if(tmp!=0):
                tmp=-tmp*math.log2(tmp)
            summary+=tmp
        return summary
    
    def compute_classifier(self):
        if self.depth>MAX_DEPTH:
            return None
        gain_r=np.empty(0)
        gain=np.empty(0)
        classifiers=[]
        for clsprop in self.clsproperties:
            if clsprop.type==0:
                _gain=self.ent_gain(clsprop)
                gain=np.append(gain,_gain)
                gain_r=np.append(gain_r,_gain/self.IV(clsprop))
                classifiers.append(clsprop)
            else:
                sorted_data=self.data.properties[clsprop.name].sorted()
                chosen=ClsProperty(clsprop.name,1)
                tmp=ClsProperty(clsprop.name,1)
                maximum=0
                for i in range(len(sorted_data)-1):
                    tmp.thresh=(sorted_data[i]+sorted_data[i+1])/2
                    new_gain=self.ent_gain(tmp)
                    if maximum<new_gain:
                        chosen=tmp.copy()
                        _gain_r=new_gain/self.IV(tmp)
                        maximum=new_gain
                gain=np.append(gain,maximum)
                classifiers.append(chosen)
                gain_r=np.append(gain_r,_gain_r)
        if len(classifiers)==0:
            return None
        m=gain.mean()
        i=0
        for i in range(len(gain)):
            if(i>=m):
                break
        return classifiers[np.argmax(gain_r[i:])]
    
    def train(self):
        if self.leaf==1:
            return
        if len(np.unique(self.data.properties['class']))==1:
            self.leaf=1
            self.result=self.data.properties['class'][0]
            return
        if self.clsproperties==None:
            self.leaf=1
            self.result=self.most_class()
            return
        self.classifier=self.compute_classifier()
        if self.classifier==None:
            self.leaf=1
            self.result=self.most_class()
            return
        for value in self.classifier.values:
            mask=(self.data.properties[self.classifier.name]==value)
            properties={}
            for key in self.data.properties.keys():
                properties[key]=self.data.properties[key][mask]
            new_data=Data(self.data.type,properties)
            if(new_data.count==0):
                new_data.leaf=1
                child.append(Node(self.classes,new_data,self.depth+1,self.ancestor))
                child[-1].result=self.most_class()
                return
            else:
                _clsproperties=self.clsproperties.copy()
                if self.classifier.type==0:
                    _clsproperties.remove(self.classifier)
                self.child.append(Node(self.classes,new_data,self.depth+1,self.ancestor,_clsproperties))
        for _child in self.child:
            _child.train()
        return
                
                
class DTree:
    def __init__(self,train_data,test_data,classes):
        self.train_data=train_data
        self.test_data=test_data
        self.classes=classes
        self.ancestor=Node(classes,data,0)
        self.ancestor.test_data=test_data
        
node=Node(np.array([0,1]),Data(np.array([0]),{'class':np.array([1,1,0,0,0,0,1,1,1]),'ccc':np.array([1,2,2,1,2,2,2,1,1])}),0)
node.train()
print(node.predict({'ccc':1}))
