import csv
import pandas as pd
import numpy as np
import math
from sklearn import svm
import time
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn import metrics
from ggplot import *





#-------------------------------------------------Splitting the data------------------------------------------------------------


traindata = []
trainlabel=[]

f = open( 'bank.csv', 'rU' ) #open the file in read universal mode
for line in f:
    cells = line.split( ";" )
    
    traindata.append( ( cells[0:len(cells)-1 ] ) )
    trainlabel.append(cells[len(cells)-1].rstrip('\n'))   #removing '\n'




#print (traindata)
#print (trainlabel)


np.savetxt("banktraindata.csv", traindata, fmt='%s', delimiter=",")		 
np.savetxt("banktrainlabel.csv", np.array(trainlabel).reshape(-1,1), fmt='%s', delimiter=",")		 

f.close()

testdata = []
testlabel=[]

f = open( 'bank-full.csv', 'rU' ) #open the file in read universal mode
for line in f:
    cells = line.split( ";" )
    
    testdata.append( ( cells[0:len(cells)-1 ] ) ) 
    testlabel.append(cells[len(cells)-1].rstrip('\n'))




#print (traindata)
#print (trainlabel)

np.savetxt("banktestdata.csv", testdata, fmt= '%s', delimiter=",")		 
np.savetxt("banktestlabel.csv", np.array(testlabel).reshape(-1,1), fmt='%s', delimiter=",")	




#---------------------------------Preprocessing the data----------------------------------------------------------------


#funtion
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def preprocess_label(file_name):
	mycsvans = open(file_name, "rb")
	csvAns = csv.reader(mycsvans)		#list of lists..csv reader
	headerans=next(csvAns, None)				#first row of Ans csv ...skip the header line
	ans=[]
	for row in csvAns:  
		#print row								
		if row[0]=='yes' :
			ans.append(1)
		elif row[0]=='no' :
			ans.append(0)
	
	#print ans
	return ans


y_1=preprocess_label("banktestlabel.csv")
y=preprocess_label("banktrainlabel.csv")









def preprocess_data(file_name,y_labels):
	mycsvfile = open(file_name, "rb")
	csvFile = csv.reader(mycsvfile)



	header=next(csvFile, None)				#first row of category names
	second_row=next(csvFile, None)				#second row to check values

	column=[]						#stores all column names
	categorical= []						#stores all categorical volumn names	
	non_categorical= []					# ""   	""	non categorical	""


	i=0
	for index in range(len(second_row)):			#loop to check which are categorical and which non categorical
		i=i+1
		column.append( header[index])
		if not is_number(second_row[index]): 
			categorical.append(header[index])
			
		else :
			non_categorical.append(header[index])



	mycsvfile.seek(1)					#getting reader back to first row(excluding header row)



	mydictionary={}						#dictionary to map all column names to its 





	for index in range(len(column)):
		#print column[index]
		mydictionary.update({column[index]:[]})

	#print range(len(column))

	for row in csvFile:
		#print row					#loop at add to dictionary
		for index in range(len(column)):
			mydictionary[column[index]].append(row[index])

	df = pd.DataFrame(mydictionary, columns = column)	#convert to dataframe all columns
	#print df


	df_new = pd.DataFrame(mydictionary, columns = non_categorical)	#data frame stores all non categorical directly
	#print df_new
	print "Binarizing features..."

	for index in range(len(categorical)):			# loop to append to data frame the categorical values after converting to numeric
		temp=pd.get_dummies(df[categorical[index]])
		df_new = pd.concat([df_new,temp], axis=1)



	matrix=df_new.as_matrix()				#convert frame to matrix


	A=np.squeeze(np.asarray(matrix))			#makes matrix to array
	A=A.astype(int)						#converts all values from strings to int
 
	min_ofeachcol=A.min(0)					#min of each column
	max_ofeachcol=A.max(0)					#max of each column

	print "Rescaling features values..."

	A=(2.0*A - max_ofeachcol - min_ofeachcol)/(max_ofeachcol - min_ofeachcol) #rescaling step aboth COntinuos and non continuos					

	mean_ofeachcol=A.mean(0)
	std_ofeachcol=A.std(0)
	print "Standardizing feature values..."
	A=(1.0*A-mean_ofeachcol)/std_ofeachcol


	i=0
	X= []
	
	for each in y_labels:
		X.append(A[i+1])
		#y.append(each)
		#each = np.append(A[i+1],each)
		#B.append(each)	
		i=i+1

	return X


X=preprocess_data("banktraindata.csv",y)
X_1=preprocess_data("banktestdata.csv",y_1)

#----------------------------------------To change the size of training data----------------
#X1=X[0:len(X)/<scale factor>]
#y1=y[0:len(y)/<scale factor>]



#-------------------------- Running SVM Classifier-----------------------------------------------------

print "Running SVM Classifier..."

#---------------------Feature Selection with c and gamma-------------------------------------------
# Here we use c=1 and gamma=auto
# After the cross validation section is executed, we can input the values of c and gamma to get accuracy
# The classifier is running on RBF kernel currently.
# TO run the linear classifier , comment the Rbf kernel classifier command and uncomment the linear kernel



#svc = svm.SVC(kernel='linear', C=1)

svc = svm.SVC(C=8192, kernel='rbf', degree=2, gamma=0.00048828125, coef0=0.0, shrinking=True, probability=True,tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)



clf=svc	

clf.fit(X, y)

#print clf.named_steps['feature_selection'].get_support()	#prints the support vectors

#-------ROC CURVE---------------

#clf.probability=True

preds = clf.predict_proba(X_1)[:,1]
fpr, tpr, _ = metrics.roc_curve(y_1, preds)
df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
graph= ggplot(df, aes(x='fpr', y='tpr')) +  geom_line( color="red" ,size =3) +  geom_abline(linetype='dashed')
print graph


#AUC Curve

auc = metrics.auc(fpr,tpr)
print auc
#----------------------------




print ""
print "Accuracy is"
print clf.score(X_1, y_1)
print ""



#-------------------------------Precision and Recall----------------------------
print "Calculating Precision and Recall..."
y_2=clf.predict(X_1)
y_3=np.array(y_1)


false_pos=0
false_neg=0
true_pos=0
true_neg=0

i =0
for eachrow in y_3:
	if eachrow !=y_2[i]:
		if y_2[i] ==1:
			false_pos=false_pos+1
		else:
			false_neg=false_neg+1
	else:
		if y_2[i]==1:
			true_pos=true_pos+1
		else:	
			true_neg=true_neg+1
	i=i+1		



if (true_pos+false_pos)==0:
	print "Precision is not defined in this case"
else :
	precision =(1.0*true_pos)/(true_pos+false_pos)
	print precision


if (true_pos+false_neg)==0:
	print "Recall is not defined in this case"
else :
	recall=(1.0*true_pos)/(true_pos+false_neg)
	print recall


print ""


#To run without cross validation, uncomment the entire following code till the end
"""
print ""
print "Running Cross Validation with Grid Search...(***This may take some time***)"


#------------------------ Cross Validation with Grid Search---------------------


#	Generates K (training, validation) pairs from the items in X.

#	Each pair is a partition of X, where validation is an iterable
#	of length len(X)/K. So each training iterable is of length (K-1)*len(X)/K.

#	If random is true, a copy of X is shuffled before partitioning,
#	otherwise its order is preserved in training and validation.



def k_fold_crossval(X, Y, K, random = False):
	if random: from random import shuffle; X=list(X); shuffle(X)
	for k in xrange(K):
		training = [x for i, x in enumerate(X) if i % K != k]
		train_labels = [y for i, y in enumerate(Y) if i % K != k]
		validation = [x for i, x in enumerate(X) if i % K == k]
		valid_labels = [y for i, y in enumerate(Y) if i % K == k]
		yield training, train_labels, validation, valid_labels

# ------------------Generating list of C and Gamma-------------------------------

c=-5
g=-15
c_list=[]
gamma_list=[]

print "Generating recommended exponential values for C and Gamma for Cross Validation..."
while (c!=17):	
	c_list.append(2**c)			#2^-5 <=C<= 2^15
	c=c+2

while (g!=5):						
	gamma_list.append(2**g)			#2^-15<=Gamma<=2^3
	g=g+2
	
print "C list is"
print c_list

print "Gamma list is"
print gamma_list


#gamma_list = [1]

results_size = (len(c_list), len(gamma_list))
results = np.zeros(results_size, dtype = np.float)

feature_model =  SelectFromModel(LinearSVC())


# The classifier is running on RBF kernel currently.
# To run the linear classifier , comment the Rbf kernel classifier command and uncomment the linear kernal


def my_func(c, gamma, training, train_labels, validation, valid_labels):
	
	#svc = svm.SVC(kernel='linear', C=c)	
	svc = svm.SVC(C=c, kernel='rbf', degree=2, gamma=gamma, coef0=0.0, shrinking=True, probability=False,tol=0.001, cache_size=200, 		class_weight=None, verbose=False, max_iter=-1, random_state=None)

	clf = Pipeline([
  	('feature_selection', feature_model),
  	('classification', svc)
	])

	print 'c,gamma are----'
	print c,gamma
	#clf=svc

	clf.fit(training, train_labels)

	#print clf.named_steps['feature_selection'].get_support()


	score = clf.score(validation, valid_labels)	
	print score
	
	return score
	

final = []
size = len(y)
for c_idx in range(len(c_list)):
	for gamma_idx in range(len(gamma_list)):
		total = 0  		
		for training, train_labels, validation, valid_labels in k_fold_crossval(X, y, K=2):
			#tune value of k above for cross validation
    			c = c_list[c_idx]
    			gamma = gamma_list[gamma_idx]
    
    			score = my_func(c, gamma, training, train_labels, validation, valid_labels)
			total = total + (score*len(valid_labels))	
		score = (total*1.0)/size

		results[c_idx, gamma_idx] = score
	   		


print results
max_index = np.argmax(results)

row = max_index/len(c_list)
col = max_index % len(gamma_list)

print 'Ideal C and gamma are'
print c_list[row], gamma_list[col]

print 'Accuracy with ideal C and gamma'
print np.max(results)
print "Where k is 2	"


"""





















