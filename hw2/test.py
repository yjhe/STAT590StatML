import sys
sys.path
sys.path.append("/home/he72/STAT590/hw2") 
import Solution as sl
import numpy as np
learner = sl.perceptron(10,1)
labels = []
labels.append(learner['otrain'])
labels.append(learner['ptrain'])

sys.argv=['5','2']
execfile('./Check.py')

import os
import sys
sys.path.append("/home/he72/STAT590/hw2")
import numpy as np
import Check as Ck
data = np.array(Ck.readfile("/home/he72/STAT590/hw2/train.txt"))

# get rid of ? missing data
contatr = np.array([1,2,7,10,13,14]) 
attrlist = np.array(range(len(data[0])))
for i in contatr:
    dum = data[data[:,i]!=np.array('?'),i]
    data[data[:,i]==np.array('?'),i] = np.mean(dum.astype(np.float))
    disatr = list(set(attrlist)-set(contatr))
for i in disatr:
        dum = max(set(list(data[:,i])), key=list(data[:,i]).count)
        data[data[:,i]==np.array('?'),i] = np.array(dum)
print "is there missing data: ", sum(sum(data[:,:15]==np.array('?')))
#os.system('read -p "$*" ')
# get labels
nsmp = len(data)
ori_train_label = data[:,-1];
ori_train_label_bool = np.zeros((data.shape[0],1))-1
ori_train_label_bool[ori_train_label==np.array('+')] = 1
# find threshold for each continuous var
maxnumthres = 1
thres = np.zeros((15,1))
for i in contatr:
    thres[i] = findThres(data[:,[i,-1]],maxnumthres)

alphaa = np.zeros((nsmp,1))	
iterr = 0

newdatafeatset = getfeature(thres,data[:,:15],featureset,disatr,contatr) # no label col

idx = 0
out = []
for i in range(nattr-1):
    idx = idx + len(orispalist[i])
    for j in range(len(orispalist[i])):
        out.append(list(np.array(orispalist[i][j])*orispared[idx:]))
        print 'j is : ',j
    print 'current attri is: ',i
    print 'current idx is: ', idx
    #print "get feature, current attribute is: ", i
