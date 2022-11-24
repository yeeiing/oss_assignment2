#PLEASE WRITE THE GITHUB URL BELOW!
# https://github.com/yeeiing/oss_assignment2.git

import sys # 명령인자값 받기

# import numpy as np # 기초 수학 연산 및 행렬계산
import pandas as pd # 데이터프레임 사용

from sklearn.model_selection import train_test_split # train, test 데이터 분할

from sklearn.svm import SVC # *서포트 벡터 머신 
from sklearn.tree import DecisionTreeClassifier # 결정트리
from sklearn.ensemble import RandomForestClassifier # *랜덤포레스트

from sklearn.metrics import accuracy_score, precision_score, recall_score 

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def load_dataset(dataset_path):
	#To-Do: Implement this function
	# dataset_path = "./heart.csv"
	df = pd.read_csv(dataset_path)
	# print(df.shape)
	# print(df.head()) # 위의 다섯줄만 임의 추출
	
	return df # data_df로 저장됨

def dataset_stat(dataset_df):	
	#To-Do: Implement this function
	
	# print(df.keys()) # print(df.columns)
	
	n_feats = len(dataset_df.keys())-1
	n_class0 = 0
	n_class1 = 0

	for i in range(len(dataset_df['target'])):
		if dataset_df['target'][i] == 0:
			n_class0+=1
		elif dataset_df['target'][i] == 1:
			n_class1+=1

	# print(numOfClass0, numOfClass1)

	# print("Number of features: ", n_feats) # number of features / 13
	# print("Number of data for class 0 : ", n_class0) # Number of data for class 0 / 499
	# print("Number of data for class 1 : ", n_class1) # Number of data for class 1 / 526

	return n_feats, n_class0, n_class1

def split_dataset(dataset_df, testset_size):
	#To-Do: Implement this function
	x = dataset_df.drop(columns="target", axis = 1)
	y = dataset_df["target"]

	# testset_size = 0.4

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = testset_size)
	# print(x_train.shape, x_test.shape)
	# print(y_train.shape, y_test.shape)

	return x_train, x_test, y_train, y_test

def decision_tree_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	dt_cls = DecisionTreeClassifier()
	dt_cls.fit(x_train, y_train)

	# print("Accuracy : ", accuracy_score(dt_cls.predict(x_test), y_test)) # Accuracy
	# print("Precision : ", precision_score(dt_cls.predict(x_test), y_test)) # Precision
	# print("Recall : ", recall_score(dt_cls.predict(x_test), y_test)) # Recall

	acc = accuracy_score(dt_cls.predict(x_test), y_test)
	prec = precision_score(dt_cls.predict(x_test), y_test)
	recall = recall_score(dt_cls.predict(x_test), y_test)

	return acc, prec, recall

def random_forest_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	rf_cls = RandomForestClassifier()
	rf_cls.fit(x_train, y_train)

	# print("Accuracy : ", accuracy_score(rf_cls.predict(x_test), y_test)) # Accuracy
	# print("Precision : ", precision_score(rf_cls.predict(x_test), y_test)) # Precision
	# print("Recall : ", recall_score(rf_cls.predict(x_test), y_test)) # Recall

	acc = accuracy_score(rf_cls.predict(x_test), y_test)
	prec = precision_score(rf_cls.predict(x_test), y_test)
	recall = recall_score(rf_cls.predict(x_test), y_test)

	return acc, prec, recall

def svm_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	# svm pipe
	svm_pipe = make_pipeline(
    	StandardScaler(),
    	SVC()
	)
	svm_pipe.fit(x_train, y_train)
	
	# print("Accuracy : ", accuracy_score(y_test, svm_pipe.predict(x_test))) # Accuracy
	# print("Precision : ", precision_score(y_test, svm_pipe.predict(x_test))) # Precision
	# print("Recall : ", recall_score(y_test, svm_pipe.predict(x_test))) # Recall

	acc = accuracy_score(y_test, svm_pipe.predict(x_test))
	prec = precision_score(y_test, svm_pipe.predict(x_test))
	recall = recall_score(y_test, svm_pipe.predict(x_test))

	return acc, prec, recall

def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)