import numpy as np

def GD(maxiter, regularization, stepSize, lamb, featureset):
	#TODO: Your code starts from here.
	#      This function should return a list of labels.
	#      e.g.:
	#	labels = [['+','-','+'],['+','+','-'],['-','+'],['+','+']]
	#	return labels
	#	where:
	#		labels[0] = original_training_labels
	#		labels[1] = prediected_training_labels
	#		labels[2] = original_testing_labels
	#		labels[3] = predicted_testing_labels
	import os
	import sys
        sys.path.append(".")
	import numpy as np
	import Check as Ck
	data = np.array(Ck.readfile("./train.txt"))

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
			
	# set up new feature space
	newdatafeatset = getfeature(thres,data[:,:15],featureset,disatr,contatr) # no label col
	w = np.zeros((newdatafeatset.shape[1],1))
	b = 0
	los = 0

	# iteration starts
	iterr = 0
	while iterr<maxiter:
		los = 0
		totloss = 0
		g = np.zeros((newdatafeatset.shape[1],1))  # initialize gradient of weigths and bias
		gb = 0
		for inloop in range(newdatafeatset.shape[0]):
			cursmp = newdatafeatset[inloop]
			cury = ori_train_label_bool[inloop]
			if cury*(np.dot(w.T,cursmp)+b) <= 1:
				g = g + cury*cursmp.reshape((cursmp.shape[0],1))
				gb = gb + cury
				los = los + 1-(cury*(np.dot(w.T,cursmp)+b))
		if regularization == 2:
			g = g - lamb*w  # add in L2 regularization term
		else:
			g = g - lamb*np.sign(w)  # add in L1 regularization term
		w = w + stepSize*g  # update weights
		b = b + stepSize*gb  # update bias
		totloss = np.dot(w.T,w) + los
		print 'iteration number: {0}, loss is {1}'.format(iterr, totloss)
		iterr = iterr + 1
	ptrain = np.sign(np.dot(newdatafeatset,w) + b)
	return {'w':w, 'b':b, 'thres':thres, 'y':ori_train_label_bool, 'X':newdatafeatset, 'otrain':ori_train_label_bool, 'ptrain': ptrain}



def calEnt(p):
	# #p is number of +,- in each value (rows) of attribute i
	#
	#
	#
	import numpy as np
	n = len(p)
	maxE = -np.log(1.0/2)
	if n>2:		
		perc = p*1.0/np.transpose(np.tile(np.sum(p,axis=1),(2,1)))
		nowE = -np.sum(perc*np.log(perc),axis=1)
		nowE[np.isnan(nowE)] = 0.0
	else:
		perc = p*1.0/np.sum(p)
		nowE = -np.sum(perc*np.log(perc))
		if np.isnan(nowE):
			nowE = 0.0
	return nowE/maxE

def findThres(dat, maxnumthres):
	# # dat is a continuous attribute value set  and associated original label, [nsmp, 2]
	#
	#
	import numpy as np
	perm = dat[dat[:,0]!=np.array('?'),:] # '?' is now dropped
	perm = perm[np.argsort(perm[:,0].astype(np.float)),:] 
	nsmp = np.shape(perm)[0]
	totthres = []
	for i in range(np.shape(perm)[0]-1):
		smp = perm[[i,i+1],1]
		if len(np.unique(smp))==2:
			totthres.append(np.mean(perm[[i,i+1],0].astype(np.float)))
	totthres = np.array(totthres)
	totthres = np.unique(totthres)
	if maxnumthres<=len(totthres):
		Ent = np.zeros((len(totthres),1))
		n = 0
		for j in totthres:
	  		pos1 = sum(perm[perm[:,1]==np.array('+'),0].astype(np.float)>=j)
	  		pos2 = sum(perm[perm[:,1]==np.array('+'),0].astype(np.float)<j)
			neg1 = sum(perm[perm[:,1]==np.array('-'),0].astype(np.float)>=j)
			neg2 = sum(perm[perm[:,1]==np.array('-'),0].astype(np.float)<j)
			dumtot = pos1+pos2+neg1+neg2
			Ent[n] = (pos1+neg1)/dumtot*calEnt(np.array([pos1,neg1]))+\
			    (pos2+neg2)/dumtot*calEnt(np.array([pos2,neg2]))
			n = n + 1
		thres = totthres[np.argsort(Ent,axis=0)[:maxnumthres]]
	else:
		thres = totthres	
	return thres.flatten()

def getfeature(thres,smp,featureset,disatr,contatr):
	# rows of smp are samples, cols are features; no label col
	# thres: vector of 15 elements with thres value for continuous variables
	# return data type is np.array
	import numpy as np
	if featureset == 1:
		orispa = getfeatinorispace(thres,smp,disatr,contatr)
		return orispa
	elif featureset == 2:
		if len(smp.shape)==1:
			out = []
			orispalist = getfeatinorispace(thres,smp,disatr,contatr,0)
			orispared = np.array(reduce(lambda x,y:x+y,orispalist))
			# get pairs
			nattr = len(orispalist)
			idx = 0
			for i in range(nattr): # loop over all attributes
				idx = idx + len(orispalist[i])
				for j in len(orispalist[i]): # loop over all values in each attribute
					out.append(np.array(orispalist[i][j])*orispared[idx:])
			return np.array(reduce(lambda x,y:x+y,out))
		else:
			outlist = []
			for nsmp in range(smp.shape[0]): # loop over all samples
				print "get feature, current sample is: ", nsmp
				out = []
				orispalist = getfeatinorispace(thres,smp[nsmp],disatr,contatr,0)
				orispared = np.array(reduce(lambda x,y:x+y,orispalist))
				# get pairs
				nattr = len(orispalist)
				idx = 0
				for i in range(nattr-1):
					idx = idx + len(orispalist[i])
					for j in range(len(orispalist[i])):
						out.append(list(np.array(orispalist[i][j])*orispared[idx:]))
					#print "get feature, current attribute is: ", i
				outlist.append(reduce(lambda x,y:x+y,out))
			return np.array(outlist)
	elif featureset == 3:
		if len(smp.shape)==1:
			out = []
			orispalist = getfeatinorispace(thres,smp,disatr,contatr,0)
			orispared = np.array(reduce(lambda x,y:x+y,orispalist))
			# get pairs
			nattr = len(orispalist)
			idx = 0
			for i in range(nattr): # loop over all attributes
				idx = idx + len(orispalist[i])
				for j in len(orispalist[i]): # loop over all values in each attribute
					out.append(np.array(orispalist[i][j])*orispared[idx:])
			outreduce = reduce(lambda x,y:x+y,out)
			togeth = reduce(lambda x,y:x+y,[list(orispared),outreduce])
			return np.array(togeth)
		else:
			outlist = []
			for nsmp in range(smp.shape[0]): # loop over all samples
				print "get feature, current sample is: ", nsmp
				out = [] # out is for the paired space
				orispalist = getfeatinorispace(thres,smp[nsmp],disatr,contatr,0)
				orispared = np.array(reduce(lambda x,y:x+y,orispalist))
				# get pairs
				nattr = len(orispalist)
				idx = 0
				for i in range(nattr-1):
					idx = idx + len(orispalist[i])
					for j in range(len(orispalist[i])):
						out.append(list(np.array(orispalist[i][j])*orispared[idx:]))
					#print "get feature, current attribute is: ", i
				outreduce = reduce(lambda x,y:x+y,out)
				togeth = reduce(lambda x,y:x+y,[list(orispared),outreduce])
				outlist.append(togeth)
			return np.array(outlist)



def sparsedot(x,y):
	import numpy as np
	# x, y are sparse matrix; x is [n * m], y can be [m,],[m * 1] or [m * p]
	n = x.shape[0]
	m = x.shape[1]
	if len(y.shape) == 1:
		# y is [m,]
		out = np.zeros((n,1))
		for i in range(n):
			out[i] = sum((x[i]==y) & (y>0))
		return out
	else: 
		# y is [m * 1] or [m * p]
		p = y.shape[1]
		out = np.zeros((n,p))
		for i in range(n):
			for j in range(p):
				out[i,j] = sum((x[i]==y[:,j]) & (y[:,j]>0))
		return out


def getfeatinorispace(thres,smp,disatr,contatr,red=1):
	# red is for weather reduce to 1 single vector for each sample; has to be off when featureset != 1
	#print 'red is : ',red
	import numpy as np
	FeaPerAttrlist = [['b','a'],\
				  [thres[1]],\
				  [thres[2]],\
				  ['u','y','l','t'],\
				  ['g','p','gg'],\
				  ['c','d','cc','i','j','k','m','r','q','w','x','e','aa','ff'],\
				  ['v','h','bb','j','n','z','dd','ff','o'],\
				  [thres[7]],\
				  ['t','f'],\
				  ['t','f'],\
				  [thres[10]],\
				  ['t','f'],\
				  ['g','p','s'],\
				  [thres[13]],\
				  [thres[14]]]
	if len(smp.shape)==1:
		feat = []
		for i in range(15):
			if i in disatr:
				feat.append([0]*len(FeaPerAttrlist[i]))
				feat[i][FeaPerAttrlist[i].index(smp[i])]=1
			elif i in contatr:
				if smp[i]>FeaPerAttrlist[i]:
					feat.append([1])
				else:
					feat.append([0])
		if red == 1:
			feat = reduce(lambda x,y:x+y,feat)
			return feat
		else:
			return feat
	else:
		out = np.zeros((smp.shape[0],47)) # 47 is total feature number
		for nsmp in range(smp.shape[0]):
			feat = []
			for i in range(15):
				if i in disatr:
						#print "current sample is ", nsmp, "feature number is ", i,  "feature is ", smp[nsmp,i]
					feat.append([0]*len(FeaPerAttrlist[i]))
					feat[i][FeaPerAttrlist[i].index(smp[nsmp,i])]=1
				elif i in contatr:
					if smp[nsmp,i]>FeaPerAttrlist[i]:
						feat.append([1])
					else:
						feat.append([0])
			out[nsmp,:] = np.array(reduce(lambda x,y:x+y,feat))
		return out
