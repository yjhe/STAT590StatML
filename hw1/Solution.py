global predlabel
predlabel = []

def DecisionTree():
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
	
	return


def DecisionTreeBounded(data, attrlist, contatr, maxDepth, maxnumthres):
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
	import sys
	sys.path
	sys.path.append("/home/he72/STAT590/hw1/data")
	import numpy as np

	nsmp = len(data)
	ori_train_label = data[:,-1];
	oriE = calEnt(np.array([sum(ori_train_label==np.array('-')),sum(ori_train_label==np.array('+'))]))

	# pick root
	Gainn = []
	templabellist = []
	for i in attrlist:
		if sum(contatr==i)==1: # if continuous attribute, find thresholds
			currentthres = findThres(data[:,[i,-1]],maxnumthres)
			thres = currentthres
			#go through each thres
			vnsmp = np.zeros((len(thres),2)) # number of smps in + , - under all attri values
			vEnt = np.zeros((len(thres),1)) # entropy for each v in attri i
			vnum = 0
			templabellist.append([i])
			for j in range(len(thres)-1): # j is thres value
				vnsmp[vnum,0] = sum(np.logical_and(data[data[:,-1]==np.array('+'),i].astype(np.float)>=thres[j],data[data[:,-1]==np.array('+'),i].astype(np.float)<=thres[j+1]))
				vnsmp[vnum,1] = sum(np.logical_and(data[data[:,-1]==np.array('-'),i].astype(np.float)>=thres[j],data[data[:,-1]==np.array('-'),i].astype(np.float)<=thres[j+1]))
				if vnsmp[vnum,0]>vnsmp[vnum,1]:
					templabellist[-1].append(j)
					templabellist[-1].append('+')
				else:
					templabellist[-1].append(j)
					templabellist[-1].append('-')
				vnum = vnum + 1
			vEnt = calEnt(vnsmp)
			Gainn.append(oriE - np.sum(vEnt*np.sum(vnsmp,axis=1)/np.sum(vnsmp)))
		else: # categerical attribute
			v = np.unique(data[:,i]) # all attri values of attri i
			vnsmp = np.zeros((len(v),2)) # number of smps in + , - under all attri values
			vEnt = np.zeros((len(v),1)) # entropy for each v in attri i
			vnum = 0
			templabellist.append([i])
			for j in v: # j is attri value
				vnsmp[vnum,0] = sum(data[data[:,-1]==np.array('+'),i]==j)
				vnsmp[vnum,1] = sum(data[data[:,-1]==np.array('-'),i]==j)
				if vnsmp[vnum,0]>vnsmp[vnum,1]:
					templabellist[-1].append(j)
					templabellist[-1].append('+')
				else:
					templabellist[-1].append(j)
					templabellist[-1].append('-')
				vnum = vnum + 1
			vEnt = calEnt(vnsmp)
			Gainn.append(oriE - np.sum(vEnt*np.sum(vnsmp,axis=1)/np.sum(vnsmp)))
	predlabel.append([templabellist[np.argmax(np.array(Gainn))]])
	# whether is a single root tree
	currentattr = attrlist[np.argmax(np.array(Gainn))]
	if max(Gainn)==oriE:
		if len(finalroots)==0:
			finalroots.append(currentattr)
		else:
			finalroots.append(currentattr)
			depth = depth + 1
		return
	# start recursive loop
	else: 
		depth = depth + 1
		newattrlist = np.delete(attrlist,currentattr)
		if len(finalroots)<depth+1:
			finalroots.append(currentattr)
		if len(finalroots)==depth+1:
			finalroots[depth].append(currentattr)		
		if sum(contatr==currentattr)==1: # if continuous attribute, find thresholds
			for j in range(len(thres)-1): # j is thres value
				DecisionTreeBounded(data, newattrlist, contatr, maxDepth, maxnumthres)
		else: # categerical attribute
			v = np.unique(data[:,currentattr]) # all attri values of attri i
			for j in v: # j is attri value
				DecisionTreeBounded(data, newattrlist, contatr, maxDepth, maxnumthres)
	return



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
	perm = perm[np.argsort(perm[:,0]),:] 
	nsmp = np.shape(perm)[0]
	#maxnumthres = 4
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
	  		pos = sum(perm[perm[:,1]==np.array('+'),0].astype(np.float)>=j)
			neg = sum(perm[perm[:,1]==np.array('-'),0].astype(np.float)<j)
			Ent[n] = calEnt(np.array([pos,neg]))
			n = n + 1
		thres = totthres[np.argsort(Ent,axis=0)[:maxnumthres]]
	else:
		thres = totthres	
	return thres.flatten()

