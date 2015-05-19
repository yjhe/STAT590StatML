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
def eval(o_train, p_train, o_valid, p_valid, o_test, p_test):
    print '\nTraining Result!'
    accuracy(o_train, p_train)
    print '\nValidation Result!'
    accuracy(o_valid, p_valid)
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
    match = sum(orig == pred)
    matchvec = orig*pred
    precision = sum(pred[matchvec==1]==1)*1.0/sum(pred==1)
    recall = sum(pred[matchvec==1]==1)*1.0/sum(orig==1)
    print '************************\nAccuracy: ' + str(float(match)/num) 
    print 'Precision: ' + str(float(precision))
    print 'Recall: ' + str(float(recall)) 
    print 'F1 score: ' + str(float(precision*recall)/(precision+recall)) + '\n***********************'




#readfile:
#   Input: filename
#   Output: return a list of rows.
def readfile(filename):    
    f = open(filename).read()
    rows = []
    for line in f.split('\r'):
        rows.append(line.split('\t'));
    return rows 


def main(maxiter,featureset):
    import sys
    print 'Number of arguments:', len(sys.argv), 'arguments.'
    print 'Argument List:', str(sys.argv)
    maxiter = int(sys.argv[-2])
    featureset = int(sys.argv[-1])
    sys.path
    sys.path.append(".") 
    import Solution as sl
    import numpy as np
    learner = sl.perceptron(maxiter,featureset)
    labels = []
    labels.append(learner['otrain'][:,0])
    labels.append(learner['ptrain'])
    match = sum(learner['otrain'][:,0] == learner['ptrain'])
    #print '*******************'
    #print 'Training accuracy is: {0:.3f}%'.format(match*1.0/len(learner['otrain'])*100.0)
    #print '*******************'

    #########################################
    ##### now run validation
    vdata = np.array(readfile("./validation.txt"))
    nvdata = vdata.shape[0]
    # get rid of ? missing data
    contatr = np.array([1,2,7,10,13,14])
    attrlist = np.array(range(len(vdata[0])))
    for i in contatr:
        dum = vdata[vdata[:,i]!=np.array('?'),i]
        vdata[vdata[:,i]==np.array('?'),i] = np.mean(dum.astype(np.float))
    disatr = list(set(attrlist)-set(contatr))
    for i in disatr:
        dum = max(set(list(vdata[:,i])), key=list(vdata[:,i]).count)
        vdata[vdata[:,i]==np.array('?'),i] = np.array(dum)
    print "is there missing data: ", sum(sum(vdata[:,:15]==np.array('?')))
    ori_v_label = vdata[:,-1];
    ori_v_label_bool = np.zeros((vdata.shape[0],1))-1
    ori_v_label_bool[ori_v_label==np.array('+')] = 1
    newdatafeatset = sl.getfeature(learner['thres'],vdata[:,:15],featureset,disatr,contatr) # no label col
    predv = np.sign(np.sum(np.tile(learner['alphaa']*learner['y'],(1,nvdata))*np.dot(learner['X'],newdatafeatset.T),axis=0))
    match = sum(predv == ori_v_label_bool[:,0])
    labels.append(ori_v_label_bool[:,0])
    labels.append(predv)
    #print '*******************'
    #print 'validation accuracy is: {0:.3f}%'.format(match*1.0/nvdata*100.0)
    #print '*******************'


    ##########################################
    ##### now run test
    tdata = np.array(readfile("./test.txt"))
    ntdata = tdata.shape[0]
    # get rid of ? missing data
    contatr = np.array([1,2,7,10,13,14])
    attrlist = np.array(range(len(tdata[0])))
    for i in contatr:
        dum = tdata[tdata[:,i]!=np.array('?'),i]
        tdata[tdata[:,i]==np.array('?'),i] = np.mean(dum.astype(np.float))
    disatr = list(set(attrlist)-set(contatr))
    for i in disatr:
        dum = max(set(list(tdata[:,i])), key=list(tdata[:,i]).count)
        tdata[tdata[:,i]==np.array('?'),i] = np.array(dum)
    print "is there missing data: ", sum(sum(tdata[:,:15]==np.array('?')))
    ori_t_label = tdata[:,-1];
    ori_t_label_bool = np.zeros((tdata.shape[0],1))-1
    ori_t_label_bool[ori_v_label==np.array('+')] = 1
    newdatafeatset = sl.getfeature(learner['thres'],tdata[:,:15],featureset,disatr,contatr) # no label col
    predt = np.sign(np.sum(np.tile(learner['alphaa']*learner['y'],(1,ntdata))*np.dot(learner['X'],newdatafeatset.T),axis=0))
    match = sum(predt == ori_t_label_bool[:,0])
    #print '*******************'
    #print 'test accuracy is: {0:.3f}%'.format(match*1.0/ntdata*100.0)
    #print '*******************'
    labels.append(ori_t_label_bool[:,0])
    labels.append(predt)

    if labels == None or len(labels) != 6:
	print '\nError: Perceptron Return Value.\n' 
    else:
	eval(labels[0],labels[1],labels[2],labels[3],labels[4],labels[5])



if __name__ == '__main__':
    import sys
    main(sys.argv[-2],sys.argv[-1])
