#import xgboost as xgb
# import xgboost as xgb
import pandas as pd
import numpy as np
import pickle
import sys
import os

from features import deepsea_features

from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.utils import resample

from imblearn.over_sampling import SMOTE

import matplotlib
matplotlib.use('TkAgg') 
from matplotlib import pyplot as plt
# from sklearn.model_selection import train_test_split

# from preprocessing import print_dataset_description
import lightgbm as lgb
# from lightgbm import LGBMClassifier

from skopt import BayesSearchCV
from skopt.space import Categorical

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
	param = {'max_depth': 60, 'num_leaves': 500, 'objective': 'binary', 'metric': ['binary_logloss']}
	
	lg = lgb.LGBMClassifier()
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

	random_state = 44

	data = pd.read_csv('../../data/data_model/data_noncoding.nctools-predictions.eqtls.deepsea_features_preprocessed.txt', sep='\t')
	
	cadd_numerical_features = get_cadd_features(data, categorical_features=False)
	cadd_all_features = get_cadd_features(data, categorical_features=True)
	# eqtl_features = get_eqtl_features()
	predictor_scores = get_predictor_scores()
	print(len(deepsea_features))

	# features = [CADD_numerical, CADD_cateogrical, CADD_all,

	runs = [
		{'features': deepsea_features, 'name': 'deepsea-features'}
		# {'features': cadd_numerical_features, 'name': 'CADD-numerical'},
		# {'features': cadd_all_features, 'name': 'CADD-all'},
		# {'features': predictor_scores, 'name': 'nc-predictors'},
		# {'features': cadd_all_features + predictor_scores, 'name': 'CADD_nc-predictors'},
		# {'features': cadd_numerical_features + ['cis_eqtls_10000bp'], 'name': 'CADD-numerical_cis-eqtls-10000bp'},
		# {'features': cadd_all_features + ['cis_eqtls_100bp'], 'name': 'CADD_cis-eqtls-100bp'},
		# {'features': cadd_all_features + predictor_scores + ['cis_eqtls_100bp'], 'name': 'CADD_nc-predictors_cis-eqtls-100bp'},
		# {'features': cadd_all_features + predictor_scores + ['cis_eqtls_100bp', 'cis_eqtls_10000bp'], 'name': 'CADD_nc-predictors_cis-eqtls-100bp-10000bp'},
		# {'features': cadd_all_features + predictor_scores + ['cis_eqtls_100bp', 'cis_eqtls_10000bp', 'trans_eqtls_500bp'], 'name': 'CADD_nc-predictors_cis-eqtls-100bp-10000bp_trans_eqtls_500bp'},
		# {'features': cadd_all_features + predictor_scores + ['cis_eqtls_100bp', 'cis_eqtls_10000bp', 'trans_eqtls_500bp', 'trans_eqtls_1000bp'], 'name': 'CADD_nc-predictors_cis-eqtls-100bp-10000bp_trans_eqtls_500bp-1000bp'}
	]

	predictions = pd.DataFrame()

	auc_scores = {}

	for x in runs:
		name = x['name']
		features = x['features']
		print(name)
		# for random_state in [53]:
		for random_state in [53, 44, 99, 12, 33]:
		# for random_state in [53, 44, 99, 12, 33, 81, 93, 61, 10, 7]:

			train, test = get_train_test_set(data, n_test_samples=150, random_state=random_state)
			X_train = train[features + ['binarized_label']]
			y_train = train['binarized_label']
			X_test = test[features]
			y_test = test['binarized_label']

			#Use oversampling on pathogenic variants.
			#Use oversampling on pathogenic variants and undersampling on benign variants.
			
			# sm = SMOTE(random_state=random_state)
			# print('Train shape before resampling')
			# print(X_train.shape)
			# print(X_train.binarized_label.value_counts())
			# print('\n')

			# print('Train shape drop NA')
			# print(X_train.dropna().shape)
			# print(X_train.dropna().binarized_label.value_counts())
			# print('\n')
			
			# X_train, y_train = sm.fit_sample(X_train.dropna(), X_train.dropna().binarized_label)
			
			# X_train = pd.DataFrame(X_train, columns=features + ['binarized_label'])
			
			# print('Train shape after resampling')
			# print(X_train.shape)
			# print(X_train.binarized_label.value_counts())
			# # print('\n')

			X_train = X_train[features]

			predictions['label' + '_' + str(random_state)] = test.label
			# print(train.shape, test.shape)
			
			if name not in auc_scores:
				auc_scores[name] = []

			

			# print('training lightgbm')
			lightgbm = train_lightgbm(X_train, y_train)
			
			# print('Time:', datetime.now() - startTime)
			y_pred_gbm = lightgbm.predict(X_test)

			predictions[name + '_' + str(random_state)] = y_pred_gbm

			auc = roc_auc_score(y_test, y_pred_gbm)
			# print('GBM ROC AUC:', auc)
			auc_scores[name].append(auc)

			lgb.plot_importance(lightgbm, figsize=(12, len(features) * 0.2))
			plt.savefig('../../figures/training/feature_importance/' + name + '_' + str(random_state) + '.pdf', bbox_inches='tight')
			plt.close('all')
			
			# print('saving model')
			# pickle.dump(model, open("../../model/test.pickle.dat", "wb"))

	for f in auc_scores:
		print(f, round(np.mean(auc_scores[f]), 3), round(np.std(auc_scores[f]), 3))
		print([round(x, 4) for x in auc_scores[f]])
		print('\n')


	# predictions.to_csv('../../data/data_model/predictions/predictions.txt', sep='\t', index=False)
	



if __name__ == '__main__':
	main()

