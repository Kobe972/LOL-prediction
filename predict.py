from decision_tree import *
from file_utils import *
types,properties=read_data(0,64000)
train_data=Data(types,properties)
types,properties=read_data(64000,80000)
test_data=Data(types,properties)
tree=DTree(train_data,test_data,np.array([0,1]))
tree.train()
print(tree.compute_accuracy())
