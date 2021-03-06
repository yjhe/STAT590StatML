'''
	Check.py is for evaluating your model. 
	Function eval() will print out the accuracy of training and testing data. 
	To call:
        	import Check
        	Check.eval(o_train, p_train, o_test, p_test)
        
	At the end of this file, it also contains how to read data from a file.
	Just for your reference.
'''

#eval:
#   Input: original training labels list, predicted training labels list,
#	       original testing labels list, predicted testing labels list.
#   Output: print out training and testing accuracy
def eval(o_train, p_train, o_test, p_test):
    print '\nTraining Result!'
    accuracy(o_train, p_train)
    print '\nTesting Result!'
    accuracy(o_test, p_test)


#accuracy:
#   Input: original labels list, predicted labels list
#   Output: print out accuracy
def accuracy(orig, pred):
    
    num = len(orig)
    if(num != len(pred)):
	print 'Error!! Num of labels are not equal.'
    	return
    match = 0
    for i in range(len(orig)):
	o_label = orig[i]
	p_label = pred[i]
	if(o_label == p_label):
	    match += 1
    print '***************\nAccuracy: ' + str(float(match)/num) + '\n***************'


#readfile:
#   Input: filename
#   Output: return a list of rows.
def readfile(filename):    
    f = open(filename).read()
    rows = []
    for line in f.split('\r'):
        rows.append(line.split('\t'));
    return rows 


if __name__ == '__main__':
    import sys
    sys.path
    sys.path.append("/home/he72/STAT590/hw1/data") 
    import Solution as sl
    import numpy as np
    global finalroots
    global thresholds
    global depth
    global contatr
    maxdepth = 10
    finalroots = []
    maxnumthres = 4
    thresholds = np.zeros((16,maxnumthres))
    depth = 0
    contatr = np.array([1,2,7,10,13,14])

    data = np.array(readfile('train.txt'))
    contatr = np.array([1,2,7,10,13,14])
    attrlist = np.array(range(len(data[0])-1))
    for i in contatr:
        dum = data[data[:,i]!=np.array('?'),i]
        data[data[:,i]==np.array('?'),i] = np.mean(dum.astype(np.float))
    categatr = list(set(attrlist)-set(contatr))
    for i in categatr:
        dum = max(set(list(data[:,i])), key=list(data[:,i]).count)
        data[data[:,i]==np.array('?'),i] = np.array(dum)


    labels = sl.DecisionTreeBounded(data, attrlist, contatr, maxdepth, maxnumthres)
    if labels == None or len(labels) != 4:
	print '\nError: DecisionTree Return Value.\n' 
    else:
	eval(labels[0],labels[1],labels[2],labels[3])
