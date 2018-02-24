import numpy as np
import math
##%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt

# function to normalise a dataset
def normalise(tempset,mean,sd,fl):
	i=0
	while i<len(tempset):
		tempset[i][0] = (tempset[i][0]-mean[0])/sd[0]
		tempset[i][1] = (tempset[i][1]-mean[1])/sd[1]
		tempset[i][2] = (tempset[i][2]-mean[2])/sd[2]
		tempset[i][3] = (tempset[i][3]-mean[3])/sd[3]
		tempset[i][4] = (tempset[i][4]-mean[4])/sd[4]
		i=i+1
	return tempset

#cost function
def costfun(trainset,m,theta):
	cf=0.0
	for i in range(1,m):
		ht = htheta(trainset[i],theta)
		cf  = cf + ( (ht-trainset[i][4])*(ht-trainset[i][4]))
	cf=cf/(2*m)	
	return cf


# calculate h(theta)
def htheta(x,t):
	ht=t[4];
	p=0
	while	p<4:
		ht =	ht + (x[p]* t[p])
		p=p+1 	
	return  ht

#rmse function
def rmse(testset,theta,sd_tst):
	res=0.0
	i=0
	n=len(testset)

	while i<n:
		ht =htheta(testset[i],theta)
		tt = (ht - testset[i][4])*sd_tst
		res = res + (tt*tt)
		i=i+1

	return math.sqrt(res/n);



# util func
def func(x,t,j,m,fl):
	q=0
	res=0
	while q<m:	
		if(j==4):
			xj=1
		else:
			xj = x[q][j]
		res = res + ( (htheta(x[q],t) - x[q][4])*xj )
		q=q+1
	return (res*1.0)/m




# linear regression function
def linear_regression(tempset,testset,precision,lamda):
	theta = np.array([0.1,0.1,0.1,0.1,0.1])
	temp = np.array([0.1,0.1,0.1,0.1,0.1])
	alpha=0.05
	m=len(tempset)


	factor=1-((lamda*alpha)/m)	


	cost=0.0
	oldcost=1
	i=0
	while (math.fabs(cost-oldcost)>precision ):

		oldcost=cost

		temp[0] = factor*theta[0] - alpha* (func( tempset,theta,0,m,0))
		temp[1] = factor*theta[1] - alpha* (func( tempset,theta,1,m,0 ))		
		temp[2] = factor*theta[2] - alpha* (func( tempset,theta,2,m,0 ))		
		temp[3] = factor*theta[3] - alpha* (func( tempset,theta,3,m,0 ))		
		temp[4] = theta[4] - alpha* (func( tempset,theta,4,m,0 ))	 	# intercept 

		theta = np.copy(temp)
		print(i,theta)
		cost = costfun(tempset,m,theta)	
		print(cost)
		#print(rmse(nor_testSet,theta,sd_tst[4]))

		i=i+1
	#print("No of Iterations: ")
	#print(i)
	return theta






######################### MAIN CODE:##############################

dataset = pd.read_csv("kc_house_data.csv")

# trim 80% as train test
trainlength = int(round( 0.8*len(dataset) ) )
testlength   = len(dataset) - trainlength
#print(trainlength,testlength)

# extract train and test sets
trainSet = dataset.head(trainlength)
testSet = dataset.tail(testlength)

#print(trainSet)

trainSet = trainSet.values
testSet = testSet.values

mean_trn = np.mean(trainSet, axis=0) 
mean_tst = np.mean(testSet, axis=0)
sd_trn = np.std(trainSet, axis=0)
sd_tst = np.std(testSet, axis=0)

print(mean_trn)
print(sd_trn)


#normalise  the training and testset
nor_trainSet = normalise(trainSet,mean_trn,sd_trn,1)
#print(nor_trainSet)
nor_testSet = normalise(testSet,mean_tst,sd_tst,0)
#print(nor_testSet)



# # call linear regrassion without regularization
theta = linear_regression(nor_trainSet,nor_testSet,0.000001,0)
# print("RMSE : ")
# print( rmse(nor_testSet,theta,sd_tst[4]))


# # # call linear regrassion with regularization
# theta_reg = linear_regression(nor_trainSet,nor_testSet,0.000001,10)	
# #print RMSE
# print("RMSE : ")
# print( rmse(nor_testSet,theta_reg,sd_tst[4]))


'''
# get rmse for diff lambda values
l=10
x=[]
y=[]
while l<=100:
	theta_reg = linear_regression(nor_trainSet,nor_testSet,0.000001,l)	
	r=rmse(nor_testSet,theta_reg,sd_tst[4])
	x.append(l)
	y.append(r)
	print(l)
	l=l+10	



plt.figure(figsize=(10,10))
plt.plot(x,y)	
plt.savefig('rmse_vs_lambda.png')
plt.show()
'''
