import pandas as pd
import numpy as np
import os
import pickle
import sys

#preprocess data

#first just scale all data and create dummy variables
#try for later: only use first 10 PC's for training the data?
#write preprocessed data to file

cadd_features = ['Ref', 'Alt', 'Type', 'Length', 'Consequence', 'GC', 'CpG', 'motifECount', 'motifEHIPos', 'motifEScoreChng', 'oAA', 'nAA', 'cDNApos', 'relcDNApos', 'CDSpos', 'relCDSpos', 'protPos', 'relProtPos', 'Domain', 'Dst2Splice', 'Dst2SplType', 'minDistTSS', 'minDistTSE', 'SIFTcat', 'SIFTval', 'PolyPhenCat', 'PolyPhenVal', 'priPhCons', 'mamPhCons', 'verPhCons', 'priPhyloP', 'mamPhyloP', 'verPhyloP', 'bStatistic', 'targetScan', 'mirSVR-Score', 'mirSVR-E', 'mirSVR-Aln', 'cHmmTssA', 'cHmmTssAFlnk', 'cHmmTxFlnk', 'cHmmTx', 'cHmmTxWk', 'cHmmEnhG', 'cHmmEnh', 'cHmmZnfRpts', 'cHmmHet', 'cHmmTssBiv', 'cHmmBivFlnk', 'cHmmEnhBiv', 'cHmmReprPC', 'cHmmReprPCWk', 'cHmmQuies', 'GerpRS', 'GerpRSpval', 'GerpN', 'GerpS', 'TFBS', 'TFBSPeaks', 'TFBSPeaksMax', 'tOverlapMotifs', 'motifDist', 'Segway', 'EncH3K27Ac', 'EncH3K4Me1', 'EncH3K4Me3', 'EncExp', 'EncNucleo', 'EncOCC', 'EncOCCombPVal', 'EncOCDNasePVal', 'EncOCFairePVal', 'EncOCpolIIPVal', 'EncOCctcfPVal', 'EncOCmycPVal', 'EncOCDNaseSig', 'EncOCFaireSig', 'EncOCpolIISig', 'EncOCctcfSig', 'EncOCmycSig', 'Grantham', 'Dist2Mutation', 'Freq100bp', 'Rare100bp', 'Sngl100bp', 'Freq1000bp', 'Rare1000bp', 'Sngl1000bp', 'Freq10000bp', 'Rare10000bp', 'Sngl10000bp', 'dbscSNV-ada_score','dbscSNV-rf_score', 'RawScore', 'PHRED']
cadd_categorical_features = ['Domain', 'nAA', 'Ref', 'PolyPhenCat', 'Alt', 'oAA', 'SIFTcat', 'Segway', 'Dst2SplType', 'Type']
cadd_numerical_features = list(set(cadd_features).difference(set(cadd_categorical_features)))


def select_noncoding_variants(df):
	pass

def impute_missing_values(df):

	na_ratios_features = df[cadd_features].isnull().sum().divide(df.shape[0])
	features_to_drop = na_ratios_features[na_ratios_features > 0.2].index

	print('Dropping', len(features_to_drop), 'columns with more than 20% NA values.')
	df = df.drop(features_to_drop, axis=1)
	print('Dropping rows with NA. (TODO: change this later to imputation).')
	
	df = df.dropna(subset=set(df.columns).intersection(cadd_features))
	print('Data has', df.shape[0], 'rows and', df.shape[1], 'columns.')

	return df

	#for value in :
#		print(df[value].isna().sum()) 

def convert_categorical_features_to_dummies(df):
	print('Converting the following categorical features to dummy variables:')
	print(cadd_categorical_features)
	print('(For now, drop them)')
	df = df.drop(set(cadd_categorical_features).intersection(set(df.columns)), axis=1)
	return df

if __name__ == '__main__':
	if not (len(sys.argv) == 2 or len(sys.argv) == 3):
		print('\nusage: preprocessing.py trainset [optional: testset]')
		print('\nRuns preprocessing on given datasets. Saves the preprocessed data in the same directory as "inputfile_preprocessed.txt". Preprocessing steps: 1) converting categorical variables to dummy variables. 2) ...')
		sys.exit()

	datasets = sys.argv[1:]
	for d in datasets:
		print('\nRunning preprocessing on', d)
		df = pd.read_csv(d, sep='\t')
		print('\nData has', df.shape[0], 'rows and', df.shape[1], 'columns.')
		#print(df.columns[80], df.columns[81], df.columns[82], df.columns[83])

		print('Imputing missing values...')

		df = impute_missing_values(df)
		df = convert_categorical_features_to_dummies(df)

		print('Saving processed data...')
		df.to_csv(d[:-4] + '_preprocessed.txt', sep='\t')

	#parser = argparse.ArgumentParser()
	#parser.add_argument("--trainset", dest="trainset", type=str)
	#Input: train_set, test_set. 
	#SAVE: train_nc_preprocessed, test_nc_preprocessed.
