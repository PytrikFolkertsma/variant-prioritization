#import xgboost as xgb
import xgboost as xgb
import pandas as pd
import sys
import os
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
import matplotlib
matplotlib.use('TkAgg') 
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from datetime import datetime
startTime = datetime.now()
from preprocessing import print_dataset_description
import lightgbm as lgb
from lightgbm import LGBMClassifier
from skopt import BayesSearchCV
from skopt.space import Categorical


os.environ['KMP_DUPLICATE_LIB_OK']='True'


cadd_features = ['Ref', 'Alt', 'Type', 'Length', 'Consequence', 'GC', 'CpG', 'motifECount', 'motifEHIPos', 'motifEScoreChng', 'oAA', 'nAA', 'cDNApos', 'relcDNApos', 'CDSpos', 'relCDSpos', 'protPos', 'relProtPos', 'Domain', 'Dst2Splice', 'Dst2SplType', 'minDistTSS', 'minDistTSE', 'SIFTcat', 'SIFTval', 'PolyPhenCat', 'PolyPhenVal', 'priPhCons', 'mamPhCons', 'verPhCons', 'priPhyloP', 'mamPhyloP', 'verPhyloP', 'bStatistic', 'targetScan', 'mirSVR-Score', 'mirSVR-E', 'mirSVR-Aln', 'cHmmTssA', 'cHmmTssAFlnk', 'cHmmTxFlnk', 'cHmmTx', 'cHmmTxWk', 'cHmmEnhG', 'cHmmEnh', 'cHmmZnfRpts', 'cHmmHet', 'cHmmTssBiv', 'cHmmBivFlnk', 'cHmmEnhBiv', 'cHmmReprPC', 'cHmmReprPCWk', 'cHmmQuies', 'GerpRS', 'GerpRSpval', 'GerpN', 'GerpS', 'TFBS', 'TFBSPeaks', 'TFBSPeaksMax', 'tOverlapMotifs', 'motifDist', 'Segway', 'EncH3K27Ac', 'EncH3K4Me1', 'EncH3K4Me3', 'EncExp', 'EncNucleo', 'EncOCC', 'EncOCCombPVal', 'EncOCDNasePVal', 'EncOCFairePVal', 'EncOCpolIIPVal', 'EncOCctcfPVal', 'EncOCmycPVal', 'EncOCDNaseSig', 'EncOCFaireSig', 'EncOCpolIISig', 'EncOCctcfSig', 'EncOCmycSig', 'Grantham', 'Dist2Mutation', 'Freq100bp', 'Rare100bp', 'Sngl100bp', 'Freq1000bp', 'Rare1000bp', 'Sngl1000bp', 'Freq10000bp', 'Rare10000bp', 'Sngl10000bp', 'dbscSNV-ada_score','dbscSNV-rf_score', 'RawScore', 'PHRED']
cadd_categorical_features = ['Consequence', 'Domain', 'nAA', 'Ref', 'PolyPhenCat', 'Alt', 'oAA', 'SIFTcat', 'Segway', 'Dst2SplType', 'Type']
cadd_numerical_features = list(set(cadd_features).difference(set(cadd_categorical_features)))

def plot_results(model):
	results = model.evals_result()

	epochs = len(results['validation_0']['error'])
	x_axis = range(0, epochs)

	fig, ax = plt.subplots()
	ax.plot(x_axis, results['validation_0']['aucpr'], label='Train')
	ax.plot(x_axis, results['validation_1']['aucpr'], label='Validation')
	# ax.plot(x_axis, results['validation_2']['aucpr'], label='Test')
	ax.legend()
	plt.ylabel('AUC PR')
	plt.title('')
	plt.show()


def train_lightgbm(X_train, y_train):
	train_data = lgb.Dataset(X_train, label=y_train)
	param = {'num_leaves': 31, 'objective': 'binary', 'metric': ['aucpr', 'binary_logloss']}
	
	lg = LGBMClassifier(random_state=53)
	lgb_b = BayesSearchCV(lg,
                      {'max_depth': Categorical(np.arange(-1, 30)),
                       'num_leaves': Categorical(np.arange(15, 500)),
                       'subsample': (0.5, 1, 'uniform'),
                       'learning_rate': (0.001, 0.5, 'log-uniform'),
                       'n_estimators': Categorical(np.arange(100, 500, 20))},
                       n_iter=30, cv=5, scoring='roc_auc')

	lgb_b.fit(X_train, y_train)
	return lgb_b
	# bst = lgb.train(param, train_data)
	# return bst

def train_xgbclassifier(X_train, y_train, X_val, y_val):
	model = xgb.XGBClassifier(objective='binary:logistic', random_state=53, eval_metric=["aucpr"])	
	model.fit(X_train, y_train, 
		eval_metric=["error", "aucpr", "logloss",], 
		eval_set=[(X_train, y_train), (X_val, y_val)], 
		verbose=False
	) 
	return model

def main():
	# if (len(sys.argv) != 3):
	# 	print('usage: train_model.py trainset testset')
	# 	sys.exit()

	train = pd.read_csv('../data/train_nc_preprocessed.txt', sep='\t')
	test = pd.read_csv('../data/test_nc_preprocessed.txt', sep='\t')

	categorical_features_dummies = train.columns[[col.startswith(tuple([f + '_' for f in cadd_categorical_features])) for col in train.columns]].tolist()
	numerical_features = sorted(list(set(train.columns).intersection(cadd_numerical_features)))
	features = [*numerical_features, *categorical_features_dummies]

	X_train, X_val, y_train, y_val = train_test_split(train[features], train['binarized_label'], test_size=0.05, random_state=53, stratify=train['binarized_label'])

	# validation_data = lgb.Dataset(X_val, reference=train_data)

	print(sum(y_train == 1), 'pathogenic variants in train set.')
	print(sum(y_val == 1), 'pathogenic variants in val set.')

	
	# # model = train_xgbclassifier(X_train, y_train, X_val, y_val)
	model = train_lightgbm(X_train, y_train)

	print('Training time:', datetime.now() - startTime)

	y_pred = model.predict(test[features])
	print('Area under the ROC curve on test set:', roc_auc_score(test['label'], y_pred))


	# y_prob_val = model.predict_proba(X_val)[:, list(model.classes_).index('Pathogenic')]
	
	# 			# fpr, tpr, thresholds = roc_curve(y_val, y_prob_val, pos_label='Pathogenic')
	# ap = average_precision_score(y_val, y_prob_val, pos_label='Pathogenic')
	# 			# precision, recall, thresholds = precision_recall_curve(list(map(lambda x: 1 if x == 'Pathogenicy_val)), y_prob_val)

	# print('Average precision', ap)

	# 			# print('Accuracy on validation set:', accuracy_score(y_val, y_pred_val))
	# 			# print('Area under the ROC curve:', roc_auc_score(y_val, y_prob_val))

	# 			# y_pred_test = model.predict(X_test)
	# 			# y_prob_test = model.predict_proba(X_test)[:, list(model.classes_).index('Pathogenic')]
	# 			# print('Accuracy on test set:', accuracy_score(y_test, y_pred_test))
	# 			# print('Area under the ROC curve:', roc_auc_score(y_test, y_prob_test))


	# 			# plot_results(model)
	print('saving model')
	pickle.dump(model, open("../model/test_model_lightgbm.pickle.dat", "wb"))




if __name__ == '__main__':
	main()

