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

features = ['#Chrom', 'Allergy/Immunology/Infectious', 'Alt', 'AnnoType', 'Audiologic/Otolaryngologic', 'Biochemical', 'CCDS', 'Cardiovascular', 'ConsDetail', 'ConsScore', 'Consequence', 'CpG', 'Craniofacial', 'Dental', 'Dermatologic', 'Dist2Mutation', 'EncExp', 'EncH3K27Ac', 'EncH3K4Me1', 'EncH3K4Me3', 'EncNucleo', 'Endocrine', 'Exon', 'FeatureID', 'Freq10000bp', 'Freq1000bp', 'Freq100bp', 'GC', 'Gastrointestinal', 'GeneID', 'GeneName', 'General', 'Genitourinary', 'GerpN', 'GerpS', 'Hematologic', 'Intron', 'Length', 'Musculoskeletal', 'Neurologic', 'Obstetric', 'Oncologic', 'Ophthalmologic', 'PHRED', 'Pos', 'Pulmonary', 'Rare10000bp', 'Rare1000bp', 'Rare100bp', 'RawScore', 'Ref', 'Renal', 'Segway', 'Sngl10000bp', 'Sngl1000bp', 'Sngl100bp', 'Stars', 'Type', 'allvalid', 'bStatistic', 'binarized_label', 'cHmmBivFlnk', 'cHmmEnh', 'cHmmEnhBiv', 'cHmmEnhG', 'cHmmHet', 'cHmmQuies', 'cHmmReprPC', 'cHmmReprPCWk', 'cHmmTssA', 'cHmmTssAFlnk', 'cHmmTssBiv', 'cHmmTx', 'cHmmTxFlnk', 'cHmmTxWk', 'cHmmZnfRpts', 'chr_pos_ref_alt', 'chr_pos_ref_alt_gene', 'clinpred', 'fathmm_score', 'inClinvar', 'inClinvar1Star', 'inClinvar2Star', 'inTest', 'inVKGLInsertion', 'inheritance', 'isAR', 'isInsertion', 'isPopulation', 'isVKGL_needsFurtherCorrection', 'label', 'mamPhCons', 'mamPhyloP', 'max_AF', 'minDistTSE', 'minDistTSS', 'motifEName', 'notinTest1', 'priPhCons', 'priPhyloP', 'provean', 'revel', 'sift', 'source', 'to_be_deleted', 'verPhCons', 'verPhyloP']

def select_noncoding_variants(df):
	pass

def drop_na_cols(df, na_ratio):
	na_ratios_features = df[cadd_features].isnull().sum().divide(df.shape[0])
	features_to_drop = na_ratios_features[na_ratios_features > na_ratio].index
	print('Dropping {} columns with more than {}% NA values.'.format(len(features_to_drop), na_ratio*100))
	df = df.drop(features_to_drop, axis=1)
	return df

def drop_na_rows(df):
	#TODO: change dropping to imputation
	print('Dropping rows with NA values')
	df = df.dropna(subset=set(df.columns).intersection(cadd_features), how='any')
	return df

def get_categorical_dummy_df(df, num_vars=10):
	categorical_features = list(set(cadd_categorical_features).intersection(df.columns))
	print('Converting {} from the {} categorical features to dummy variables: {}.'.format(len(categorical_features), len(cadd_categorical_features), categorical_features))
	for feature in categorical_features:
		counts = df[feature].value_counts()
		if len(counts) > num_vars:
			features = counts.index[:num_vars]
		else:
			features = counts.index
		df[feature] = np.where(df[feature].isin(features), df[feature], "other")

	dummies = pd.get_dummies(df[categorical_features], prefix=categorical_features)
	print('Dummies DF shape:', dummies.shape)
	return dummies

def print_dataset_description(df):
	print('Data has {} rows and {} columns. {} pathogenic variants ({}%)'.format(
		df.shape[0], 
		df.shape[1], 
		df['label'].value_counts()['Pathogenic'], 
		round(df['label'].value_counts()['Pathogenic']/df.shape[0]*100, 2))
	)

def run_preprocessing(path, convert_categorical_vars=True, drop_rows=True, trainset_cols=None, is_testset=False):
	print('\nRunning preprocessing on', path)
	df = pd.read_csv(path, sep='\t')

	print_dataset_description(df)
	
	if not is_testset:
		df = drop_na_cols(df, na_ratio=0.3)

	if drop_rows:
		df = drop_na_rows(df)

	if convert_categorical_vars:
		dummies = get_categorical_dummy_df(df, num_vars=10)
		if is_testset:
			#add missing dummy variables from trainset to testset with all values set on 0.
			print('Adding missing dummy variables to testset...')
			categorical_features_train = trainset_cols[[col.startswith(tuple([f + '_' for f in cadd_categorical_features])) for col in trainset_cols]].tolist()
			categorical_dummies_not_in_testset = list(set(categorical_features_train).difference(dummies.columns))
			categorical_features_zeros_df = pd.DataFrame(np.zeros(shape=(df.shape[0], len(categorical_dummies_not_in_testset))), columns=categorical_dummies_not_in_testset, index=df.index)
			dummies = pd.concat([dummies, categorical_features_zeros_df], axis=1)
			print('Dummies DF shape:', dummies.shape)
		df = pd.concat([df, dummies], axis=1) 
	else:
		df = df.drop(list(set(cadd_categorical_features).intersection(df.columns)), axis=1)

	print_dataset_description(df)
	
	return df

def main():
	if not (len(sys.argv) == 2 or len(sys.argv) == 3):
		print('\nusage: preprocessing.py trainset testset]')
		print('\nRuns preprocessing on given datasets. Saves the preprocessed data in the same directory as "inputfile_preprocessed.txt". Preprocessing steps: 1) converting categorical variables to dummy variables. 2) ...')
		sys.exit()

	trainset = run_preprocessing(sys.argv[1], convert_categorical_vars=False, drop_rows=False)
	print('Saving trainset...')
	trainset.to_csv(sys.argv[1][:-4] + '_preprocessed.txt', sep='\t', index=False)
	testset = run_preprocessing(sys.argv[2], convert_categorical_vars=False, drop_rows=False, is_testset=True, trainset_cols=trainset.columns)
	print('Saving testset...')	
	testset.to_csv(sys.argv[2][:-4] + '_preprocessed.txt', sep='\t', index=False)


if __name__ == '__main__':
	main()

	
