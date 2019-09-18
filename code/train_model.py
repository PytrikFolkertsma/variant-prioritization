#import xgboost as xgb
from xgboost import XGBClassifier
import pandas as pd
import sys

cadd_features = ['Ref', 'Alt', 'Type', 'Length', 'Consequence', 'GC', 'CpG', 'motifECount', 'motifEHIPos', 'motifEScoreChng', 'oAA', 'nAA', 'cDNApos', 'relcDNApos', 'CDSpos', 'relCDSpos', 'protPos', 'relProtPos', 'Domain', 'Dst2Splice', 'Dst2SplType', 'minDistTSS', 'minDistTSE', 'SIFTcat', 'SIFTval', 'PolyPhenCat', 'PolyPhenVal', 'priPhCons', 'mamPhCons', 'verPhCons', 'priPhyloP', 'mamPhyloP', 'verPhyloP', 'bStatistic', 'targetScan', 'mirSVR-Score', 'mirSVR-E', 'mirSVR-Aln', 'cHmmTssA', 'cHmmTssAFlnk', 'cHmmTxFlnk', 'cHmmTx', 'cHmmTxWk', 'cHmmEnhG', 'cHmmEnh', 'cHmmZnfRpts', 'cHmmHet', 'cHmmTssBiv', 'cHmmBivFlnk', 'cHmmEnhBiv', 'cHmmReprPC', 'cHmmReprPCWk', 'cHmmQuies', 'GerpRS', 'GerpRSpval', 'GerpN', 'GerpS', 'TFBS', 'TFBSPeaks', 'TFBSPeaksMax', 'tOverlapMotifs', 'motifDist', 'Segway', 'EncH3K27Ac', 'EncH3K4Me1', 'EncH3K4Me3', 'EncExp', 'EncNucleo', 'EncOCC', 'EncOCCombPVal', 'EncOCDNasePVal', 'EncOCFairePVal', 'EncOCpolIIPVal', 'EncOCctcfPVal', 'EncOCmycPVal', 'EncOCDNaseSig', 'EncOCFaireSig', 'EncOCpolIISig', 'EncOCctcfSig', 'EncOCmycSig', 'Grantham', 'Dist2Mutation', 'Freq100bp', 'Rare100bp', 'Sngl100bp', 'Freq1000bp', 'Rare1000bp', 'Sngl1000bp', 'Freq10000bp', 'Rare10000bp', 'Sngl10000bp', 'dbscSNV-ada_score','dbscSNV-rf_score', 'RawScore', 'PHRED']
cadd_categorical_features = ['Consequence', 'Domain', 'nAA', 'Ref', 'PolyPhenCat', 'Alt', 'oAA', 'SIFTcat', 'Segway', 'Dst2SplType', 'Type']
cadd_numerical_features = list(set(cadd_features).difference(set(cadd_categorical_features)))


if __name__ == '__main__':
	if (len(sys.argv) != 3):
		print('usage: train_model.py trainset testset')
		sys.exit()

	train = pd.read_csv(sys.argv[1], sep='\t')
	test = pd.read_csv(sys.argv[2], sep='\t')
	
	#dtrain = xgb.DMatrix(train[set(train.columns).intersection(cadd_numerical_features)], label=train['label'])
	
	model = XGBClassifier()
	model.fit(train[set(train.columns).intersection(cadd_numerical_features)], train['label'])
	#dtest = xgb.DMatrix(train[set(test.columns).intersection(cadd_numerical_features)])

	#param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
	#evallist = [(dtest, 'eval'), (dtrain, 'train')]
	#num_round = 2
	
	#bst = xgb.train(param, dtrain, num_round)

	

