#import xgboost as xgb
import xgboost as xgb
import pandas as pd
import sys
import os
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error, roc_curve, roc_auc_score
import matplotlib
matplotlib.use('TkAgg') 
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
import pprint
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def main():
	# model = pickle.load(open("../model/test_model_lightgbm.pickle.dat", "rb"))
	model = pickle.load(open("../model/test_model.pickle.dat", "rb"))
	test = pd.read_csv('../data/test_nc_preprocessed.txt', sep='\t')

	# for item in model.__dict__:
	# 	print(item)
	# 	print(model.__dict__[item])
	# 	# print(model.best_estimator_.__dict__[item])
	# # print(model)

	# print(model.best_estimator_)

	features = model._Booster.feature_names

	X_test = test[features]
	y_test = test['label']
	y_pred = model.predict(X_test)
	y_prob = model.predict_proba(X_test)[:, list(model.classes_).index('Pathogenic')] #get probabilities for Pathogenic prediction
	auc_score = roc_auc_score(y_test, y_prob)
	print('AUC SCORE:', auc_score)

	fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label='Pathogenic')
	plt.plot(fpr, tpr,'r-',label = 'Pathogenic predictions')
	plt.plot([0,1],[0,1],'k-',label='random')
	plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
	plt.legend()
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.show()

if __name__ == '__main__':
	main()

