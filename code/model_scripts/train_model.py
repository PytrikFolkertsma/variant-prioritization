#import xgboost as xgb
# import xgboost as xgb
import pandas as pd
import numpy as np
import pickle
import sys
import os
import re

from features import deepsea_features
from features import p005
from features import p001
from features import p0001
from features import p00001

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.utils import resample

import imblearn.over_sampling 
import imblearn.under_sampling

import matplotlib
matplotlib.use('TkAgg') 
from matplotlib import pyplot as plt
# from sklearn.model_selection import train_test_split

# from preprocessing import print_dataset_description
import lightgbm as lgb
# from lightgbm import LGBMClassifier

from skopt import BayesSearchCV
from skopt.space import Categorical

import json

import time
from datetime import datetime
startTime = datetime.now()



os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
	param = {'max_depth': 30, 'num_leaves': 100, 'objective': 'binary', 'metric': ['binary_logloss']}
	# param = {'max_depth': 60, 'num_leaves': 500, 'objective': 'binary', 'metric': ['binary_logloss']}
	
	# lg = lgb.LGBMClassifier()

	# cv_results = lgb.cv(param, train_data, num_round=10, nfold=5)
	# print(cv_results)
	# lgb_b = BayesSearchCV(lg,
 #                      {'max_depth': Categorical(np.arange(-1, 30)),
 #                       'num_leaves': Categorical(np.arange(15, 500)),
 #                       'subsample': (0.5, 1, 'uniform'),
 #                       'learning_rate': (0.001, 0.5, 'log-uniform'),
 #                       'n_estimators': Categorical(np.arange(100, 500, 20))},
 #                       n_iter=30, cv=5, scoring='auc')

	# lgb_b.fit(X_train, y_train)
	# return lgb_b
	bst = lgb.train(param, train_data)
	return bst

def train_xgbclassifier(X_train, y_train):
	#get validation set? 
	model = xgb.XGBClassifier(objective='binary:logistic')	
	model.fit(X_train, y_train)
	# model.fit(X_train, y_train, 
	# 	eval_metric=["error", "aucpr", "logloss",], 
	# 	eval_set=[(X_train, y_train), (X_val, y_val)], 
	# 	verbose=False
	# ) 
	return model

def get_cadd_features(data, categorical_features=False):
	#returns cadd features 
	cadd_features = ['Ref', 'Alt', 'Type', 'Length', 'Consequence', 'GC', 'CpG', 'motifECount', 'motifEHIPos', 'motifEScoreChng', 'oAA', 'nAA', 'cDNApos', 'relcDNApos', 'CDSpos', 'relCDSpos', 'protPos', 'relProtPos', 'Domain', 'Dst2Splice', 'Dst2SplType', 'minDistTSS', 'minDistTSE', 'SIFTcat', 'SIFTval', 'PolyPhenCat', 'PolyPhenVal', 'priPhCons', 'mamPhCons', 'verPhCons', 'priPhyloP', 'mamPhyloP', 'verPhyloP', 'bStatistic', 'targetScan', 'mirSVR-Score', 'mirSVR-E', 'mirSVR-Aln', 'cHmmTssA', 'cHmmTssAFlnk', 'cHmmTxFlnk', 'cHmmTx', 'cHmmTxWk', 'cHmmEnhG', 'cHmmEnh', 'cHmmZnfRpts', 'cHmmHet', 'cHmmTssBiv', 'cHmmBivFlnk', 'cHmmEnhBiv', 'cHmmReprPC', 'cHmmReprPCWk', 'cHmmQuies', 'GerpRS', 'GerpRSpval', 'GerpN', 'GerpS', 'TFBS', 'TFBSPeaks', 'TFBSPeaksMax', 'tOverlapMotifs', 'motifDist', 'Segway', 'EncH3K27Ac', 'EncH3K4Me1', 'EncH3K4Me3', 'EncExp', 'EncNucleo', 'EncOCC', 'EncOCCombPVal', 'EncOCDNasePVal', 'EncOCFairePVal', 'EncOCpolIIPVal', 'EncOCctcfPVal', 'EncOCmycPVal', 'EncOCDNaseSig', 'EncOCFaireSig', 'EncOCpolIISig', 'EncOCctcfSig', 'EncOCmycSig', 'Grantham', 'Dist2Mutation', 'Freq100bp', 'Rare100bp', 'Sngl100bp', 'Freq1000bp', 'Rare1000bp', 'Sngl1000bp', 'Freq10000bp', 'Rare10000bp', 'Sngl10000bp', 'dbscSNV-ada_score','dbscSNV-rf_score']
	cadd_categorical_features = ['Consequence', 'Domain', 'nAA', 'Ref', 'PolyPhenCat', 'Alt', 'oAA', 'SIFTcat', 'Segway', 'Dst2SplType', 'Type']
	cadd_numerical_features = list(set(cadd_features).difference(set(cadd_categorical_features)))
	categorical_features_dummies = data.columns[[col.startswith(tuple([f + '_' for f in cadd_categorical_features])) for col in data.columns]].tolist()
	numerical_features = sorted(list(set(data.columns).intersection(cadd_numerical_features)))
	if categorical_features:
		features = [*numerical_features, *categorical_features_dummies]
	else:
		features = numerical_features
	return features

def get_deepsea_features():
	return deepsea_features

def get_predictor_scores():
	return([
		'RawScore',
		'ReMM_score',
		'DeepSEA_Functional_significance_score',
		'Eigen_raw',
		'Eigen PC_raw',
		'LINSIGHT_score',
	])

def get_eqtl_features():
	#search for eqtl column in data columns
	return([
		'cis_eqtls_100bp', 
		'cis_eqtls_250bp', 
		'cis_eqtls_500bp'
	]) 


def get_train_test_set(data, n_test_samples, random_state):
	data_hc = data[data.sample_weight == 1]
	test = pd.concat([
		resample(data_hc[data_hc.label == 'Pathogenic'], replace=False, n_samples = n_test_samples, random_state=random_state),
		resample(data_hc[data_hc.label == 'Benign'], replace=False, n_samples = n_test_samples, random_state=random_state)
	])
	train = data.drop(test.index)
	return train, test


def run_featuresets(data, random_state, skf, featuresets):
	
	results = {}
	count = 1

	for train_index, test_index in skf.split(data, data['binarized_label']):
		print('FOLD', count)
		
		

		for run in featuresets:
			features = run['features']
			name = run['name']
			print(name)
			
			print('train test shape')
			train = data.loc[train_index, ].dropna(subset=features).reset_index(drop=True)
			test = data.loc[test_index, ].dropna(subset=features).reset_index(drop=True)
			print(train.shape, test.shape)

			X_train, X_test = train[features], test[features]
			y_train, y_test = train['binarized_label'], test['binarized_label']

			results = train_return_performance(X_train, y_train, X_test, y_test, results, name, random_state)


		print('\n')

	return results


def train_return_performance(X_train, y_train, X_test, y_test, results, name, random_state):
	lightgbm = train_lightgbm(X_train, y_train)
	y_pred_all = lightgbm.predict(X_test)
	undersampling = imblearn.under_sampling.RandomUnderSampler(random_state=random_state)
	X_test_res, y_test_res = undersampling.fit_resample(X_test, y_test)
	y_pred_balanced = y_pred_all[undersampling.sample_indices_]	

	if name not in results:
		results[name] = {'AP': [], 'AUC_all': [], 'AUC_balanced': []}

	results[name]['AP'].append(average_precision_score(y_test, y_pred_all))
	results[name]['AUC_all'].append(roc_auc_score(y_test, y_pred_all))
	results[name]['AUC_balanced'].append(roc_auc_score(y_test_res, y_pred_balanced))
	
	return results

def run_oversampling_techniques(data, features, random_state, skf, training_strategies):
	#1. Find out which oversampling technique works best. 
	#. Oversample SMOTE, sampling_strategy=0.2
	#  Oversample SMOTE, sampling_strategy=0.5
	results = {}
	fold = 1
	print(data.shape)
	data = data.dropna(subset=features).reset_index(drop=True)
	print(data.shape)
	for train_index, test_index in skf.split(data, data['binarized_label']):
		print('FOLD', fold)
		fold += 1
		train = data.loc[train_index, ]
		test = data.loc[test_index, ]
		print(train.shape, test.shape)
		for oversampling in training_strategies:
			print('OVERSAMPLING:', str(oversampling).split('(')[0])
			if oversampling == None:
				results = train_return_performance(train[features], train['binarized_label'], test[features], test['binarized_label'], results=results, name='no_oversampling', random_state=random_state)
			else:
				X_train, y_train = oversampling.fit_resample(train[features], np.array(train['binarized_label'], dtype=np.int32))
				X_test, y_test = test[features], test['binarized_label']
				name = str(oversampling).split('(')[0]
				parameters = list(filter(re.compile('k_neighbors|n_neighbors|m_neighbors').findall, re.split(r'\(|\)|,', str(oversampling))))
				print('>>>NAME')
				print(name + ', ' + ', '.join(parameters))
				results = train_return_performance(X_train, y_train, X_test, y_test, results=results, name=name + ', ' + ', '.join(parameters), random_state=random_state)
				print('\n')

	return results

def main():
	# if (len(sys.argv) != 3):
	# 	print('usage: train_model.py trainset testset')
	# 	sys.exit()

	#Train on:

	#1. 5 times: Select balanced test set from the data (high-confidence). Use other variants as train set.
	#2. Train on:
	# 	- CADD scores
	# 	- Predictor scores
	# 	- Whole dataset / HC / 
	# 	- Oversampling/undersampling
	#3. Make predictions on test set
	#4. Write results to file called [train/testset features_used-model sampling-technique trained-on-HC, ...].
	#.   Open file with predictions, add column -> save
	#5. Integrate all predictions in one file
	#6. Evaluate results. Show average ROC AUC

	random_state = 0

	data = pd.read_csv('../../data/data_model/data_noncoding.nctools-predictions.eqtls.deepsea_features_preprocessed.txt', sep='\t')
	
	cadd_numerical_features = get_cadd_features(data, categorical_features=False)
	cadd_all_features = get_cadd_features(data, categorical_features=True)
	# eqtl_features = get_eqtl_features()
	predictor_scores = get_predictor_scores()
	# print(len(deepsea_features))


	runs = [
		# {'features': p005, 'name': 'Pval 0.05'},
		# {'features': p001, 'name': 'Pval 0.01'},
		# {'features': p0001, 'name': 'Pval 0.001'},
		# {'features': p00001, 'name': 'Pval 0.0001'},
		# {'features': p005 + ['SK-N-SH_RA|p300|None'], 'name': 'Pval 0.05 + D1'},
		# {'features': p001 + ['SK-N-SH_RA|p300|None'], 'name': 'Pval 0.01 + D1'},
		# {'features': p0001 + ['SK-N-SH_RA|p300|None'], 'name': 'Pval 0.001' + 'D1'},
		# {'features': p00001 + ['SK-N-SH_RA|p300|None'], 'name': 'Pval 0.0001' + 'D1'},
		{'features': deepsea_features, 'name': 'deepsea-features'},
		# {'features': cadd_numerical_features, 'name': 'CADD-numerical'},
		# {'features': cadd_numerical_features + ['SK-N-SH_RA|p300|None'], 'name': 'CADD-numerical_deepSEA-1'},
		# {'features': cadd_numerical_features + ['SK-N-SH_RA|p300|None', 'SH-SY5Y|GATA3|None', 'SH-SY5Y|GATA-2|None', 'HeLa-S3|ZZZ3|None', 'SK-N-SH_RA|DNase|None'], 'name': 'CADD-numerical_deepSEA-5'},
		# {'features': cadd_numerical_features + ['SK-N-SH_RA|p300|None', 'SH-SY5Y|GATA3|None', 'SH-SY5Y|GATA-2|None', 'HeLa-S3|ZZZ3|None', 'SK-N-SH_RA|DNase|None', 'PFSK-1|Sin3Ak-20|None', 'SK-N-SH_RA|YY1|None', 'Monocytes-CD14+RO01746\xa0|H3K27me3|None', 'Dnd41|EZH2|None', 'NHDF-Ad|H2AZ|None'], 'name': 'CADD-numerical_deepSEA-10'},
		# {'features': cadd_all_features, 'name': 'CADD-all'},
		# {'features': predictor_scores, 'name': 'nc-predictors'},
		# {'features': cadd_all_features + predictor_scores, 'name': 'CADD_nc-predictors'},
		# {'features': cadd_numerical_features + predictor_scores, 'name': 'CADD-numerical_nc-predictors'},
		# {'features': cadd_numerical_features + ['cis_eqtls_10000bp'], 'name': 'CADD-numerical_cis-eqtls-10000bp'},
		# {'features': cadd_all_features + ['cis_eqtls_10000bp'], 'name': 'CADD_cis-eqtls-10000bp'},
		# {'features': cadd_all_features + ['cis_eqtls_10000bp'] + predictor_scores, 'name': 'CADD_cis-eqtls-10000bp_nc-predictors'},
		# {'features': cadd_all_features + ['cis_eqtls_10000bp'] + predictor_scores + ['SK-N-SH_RA|p300|None', 'SH-SY5Y|GATA3|None', 'SH-SY5Y|GATA-2|None', 'HeLa-S3|ZZZ3|None', 'SK-N-SH_RA|DNase|None'], 'name': 'CADD_cis-eqtls-10000bp_nc-predictors_deepSEA-5'},
	]

	sampling_strategy = 0.2

	training_strategies = [
		None,

		imblearn.over_sampling.SMOTE(random_state=random_state, sampling_strategy=sampling_strategy, k_neighbors=5), #default
		imblearn.over_sampling.SMOTE(random_state=random_state, sampling_strategy=sampling_strategy, k_neighbors=15),
		imblearn.over_sampling.SMOTE(random_state=random_state, sampling_strategy=sampling_strategy, k_neighbors=40),
		
		imblearn.over_sampling.BorderlineSMOTE(random_state=random_state, sampling_strategy=sampling_strategy, k_neighbors=5, m_neighbors=10), #deafult
		imblearn.over_sampling.BorderlineSMOTE(random_state=random_state, sampling_strategy=sampling_strategy, k_neighbors=5, m_neighbors=25),
		imblearn.over_sampling.BorderlineSMOTE(random_state=random_state, sampling_strategy=sampling_strategy, k_neighbors=5, m_neighbors=50),
		imblearn.over_sampling.BorderlineSMOTE(random_state=random_state, sampling_strategy=sampling_strategy, k_neighbors=15, m_neighbors=25),
		imblearn.over_sampling.BorderlineSMOTE(random_state=random_state, sampling_strategy=sampling_strategy, k_neighbors=40, m_neighbors=25),
		imblearn.over_sampling.BorderlineSMOTE(random_state=random_state, sampling_strategy=sampling_strategy, k_neighbors=15, m_neighbors=50),
		imblearn.over_sampling.BorderlineSMOTE(random_state=random_state, sampling_strategy=sampling_strategy, k_neighbors=40, m_neighbors=50),
		
		imblearn.over_sampling.ADASYN(random_state=random_state, sampling_strategy=sampling_strategy, n_neighbors=5), #default
		imblearn.over_sampling.ADASYN(random_state=random_state, sampling_strategy=sampling_strategy, n_neighbors=15),
		imblearn.over_sampling.ADASYN(random_state=random_state, sampling_strategy=sampling_strategy, n_neighbors=40),

		imblearn.over_sampling.SVMSMOTE(random_state=random_state, sampling_strategy=sampling_strategy, k_neighbors=5, m_neighbors=10),
		imblearn.over_sampling.SVMSMOTE(random_state=random_state, sampling_strategy=sampling_strategy, k_neighbors=5, m_neighbors=25), #default
		imblearn.over_sampling.SVMSMOTE(random_state=random_state, sampling_strategy=sampling_strategy, k_neighbors=5, m_neighbors=50),
		imblearn.over_sampling.SVMSMOTE(random_state=random_state, sampling_strategy=sampling_strategy, k_neighbors=15, m_neighbors=25),
		imblearn.over_sampling.SVMSMOTE(random_state=random_state, sampling_strategy=sampling_strategy, k_neighbors=40, m_neighbors=25),
		imblearn.over_sampling.SVMSMOTE(random_state=random_state, sampling_strategy=sampling_strategy, k_neighbors=15, m_neighbors=50),
		imblearn.over_sampling.SVMSMOTE(random_state=random_state, sampling_strategy=sampling_strategy, k_neighbors=40, m_neighbors=50),

	]

	skf = StratifiedKFold(n_splits=5, random_state=random_state, shuffle=True)

	featuresets_results = run_featuresets(data, random_state=random_state, skf=skf, featuresets=runs)
	##Write results to file
	with open('../../data/data_model/featuresets_{}_rs{}.txt'.format(time.strftime("%Y%m%d-%H%M%S"), random_state), 'w') as file:
		file.write(json.dumps(featuresets_results))

	print(time.time() - startTime)
	# features = cadd_all_features + predictor_scores + ['cis_eqtls_100bp', 'cis_eqtls_10000bp']
	# oversampling_results = run_oversampling_techniques(data, features=features, random_state=random_state, skf=skf, training_strategies=training_strategies)
	# ##Write results to file
	# with open('../../data/data_model/oversampling_all-features_{}.txt'.format(time.strftime("%Y%m%d-%H%M%S")), 'w') as file:
	# 	file.write(json.dumps(oversampling_results))


if __name__ == '__main__':
	main()

