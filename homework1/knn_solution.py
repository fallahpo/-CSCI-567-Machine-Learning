import numpy as np
import json
from itertools import groupby
#===========================Q4.1=====================================
def compute_distances(Xtrain,X):
	num_test = X.shape[0]
	num_train = Xtrain.shape[0]
	dists = np.zeros((num_test, num_train)) 
	A = np.sum(X**2,axis=1).reshape(num_test,1)
	B = np.sum(Xtrain**2,axis=1).reshape(num_train,1)
	AB = np.dot(X,np.transpose(Xtrain))
	dists = np.sqrt(-2*AB + A + np.transpose(B))
	return dists

#==========================Q4.2======================================
def predict_labels(k,ytrain,dists):
	
	num_test = dists.shape[0]
	ypred = np.zeros(num_test)
	for i in range(num_test):
		sort_list = np.argsort(dists[i])	
		closest_y = []
		for j in range(k):
			closest_y.append(ytrain[sort_list[j]])
		closest_y.sort()	
		count = [len(list(group)) for key, group in groupby(closest_y)]
		key = [key for key, group in groupby(closest_y)]
		
		max_val = max(count)
		for temp in range(len(count)):
			if count[temp] == max_val:
				ypred[i] = key[temp]
				break
		
	#print(ypred)
	return ypred

#===========================Q4.3=====================================
def compute_accuracy(y,ypred):
	cnt = 0
	for i in range(len(ypred)):
		if y[i] == ypred[i]:
			cnt+=1
	acc = float(cnt)/len(ypred)
	return acc

#==========================Q4.4=====================================
def find_best_k(K,ytrain,dists,yval):
	validation_accuracy = []
	for k in K:
		ypred = predict_labels(k,ytrain,dists)
		acc = compute_accuracy(yval,ypred)
		validation_accuracy.append(acc)
		print("The validation accuracy is",acc,"when k =",k)
	a = max(validation_accuracy)
	idx = validation_accuracy.index(a)
	best_k = K[idx]
	print(best_k)
	return best_k,validation_accuracy



#============================END=====================================

'''
Please DO NOT CHANGE ANY CODE below this line.
You should only write your code in the above functions.
'''

def data_processing(data):
	train_set, valid_set,test_set = data['train'],data['valid'],data['test']
	Xtrain = train_set[0]
	ytrain = train_set[1]
	Xval = valid_set[0]
	yval = valid_set[1]
	Xtest = test_set[0]
	ytest = test_set[1]
	
	Xtrain = np.array(Xtrain)
	Xval = np.array(Xval)
	Xtest = np.array(Xtest)
	
	ytrain = np.array(ytrain)
	yval = np.array(yval)
	ytest = np.array(ytest)
	
	return Xtrain,ytrain,Xval,yval,Xtest,ytest
	
def main():
	input_file = 'mnist_subset.json'
	output_file = 'knn_output.txt'

	with open(input_file) as json_data:
		data = json.load(json_data)
	
	#==================Compute distance matrix=======================
	K=[1,3,5,7,9]	
	
	Xtrain,ytrain,Xval,yval,Xtest,ytest = data_processing(data)
	
	dists = compute_distances(Xtrain,Xval)
	
	#===============Compute validation accuracy when k=5=============
	ypred = predict_labels(5,ytrain,dists)
	acc = compute_accuracy(yval,ypred)
	
	
	#==========select the best k by using validation set==============
	best_k,validation_accuracy = find_best_k(K,ytrain,dists,yval)

	
	#===============test the performance with your best k=============
	dists = compute_distances(Xtrain,Xtest)
	ypred = predict_labels(best_k,ytrain,dists)
	test_accuracy = compute_accuracy(ytest,ypred)
	
	#====================write your results to file===================
	f=open(output_file,'w')
	for i in range(len(K)):
		f.write('%d %.3f' % (K[i], validation_accuracy[i])+'\n')
	f.write('%s %.3f' % ('test', test_accuracy))
	f.close()
	
if __name__ == "__main__":
	main()