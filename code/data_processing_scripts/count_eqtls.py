#output: 
#columns: bp100, bp500, bp1000
#for every variant: show n eqtls within 100 bp, 500, 1000, etc. 
#
#OUTPUT:
#
# Chrom		Pos		cis_eqtl_bp100	cis_eqtl_bp500	...
#
#


#Make dictionaries per chromosome for eqtl file.
#{1: [eqtl positions], 2: [eqtl positions]} 

#Order trainset on chromsome and position.
#For every variant:
#	min_pos = pos - bp
#	max_pos = pos + bp
#	Retrieve positions list.
#	sum(filter list for values between min and max)
#	write to file

import pandas as pd
import numpy as np
import sys
from functools import reduce

def parse_eqtl_file(path):
	#returns a dict that contains a list of eqtl positions per chromosome {1: [eqtl positions], 2: [...]}
	eqtl_file = open(path).readlines()
	# eqtl_file = open('../../files/dummy_eqtls.txt').readlines()
	eqtl_dict = {}
	firstline = eqtl_file[0].split('\t')
	position_index = firstline.index('SNPPos')
	chrom_index = firstline.index('SNPChr')

	for line in eqtl_file[1:]:
		line = line.split('\t')
		pos = line[position_index]
		chrom = line[chrom_index]
		if chrom not in eqtl_dict:
			eqtl_dict[chrom] = set()
		eqtl_dict[chrom].add(int(pos))
	return eqtl_dict

def count_eqtls(data, eqtl_dict, prefix, bp=100):
	#return df: chr, pos, eqtl count
	print('counting', prefix, 'eqtls', 'within', bp, 'bp')
	rows = []
	for chromosome, df in data.groupby(by='#Chrom'):
		# print('CHR',chromosome)
		df = df.sort_values(by='Pos')
		if chromosome in eqtl_dict:
			eqtl_positions = sorted(list(eqtl_dict[chromosome]))
			starting_index = 0
			variants = np.array([df['#Chrom'].tolist(), df['Pos'].tolist()])

			for v in range(len(variants[0])):
				chrom = variants[0][v]
				pos = int(variants[1][v])
				min_pos = pos - bp 
				max_pos = pos + bp
				n_eqtls = 0

				i = starting_index

				setStartingIndex = True
				while eqtl_positions[i] <= max_pos:
					if eqtl_positions[i] >= min_pos: #when the minimum position is reached, start counting eqtls
						n_eqtls += 1
						if setStartingIndex: #set the index of the minimum position as starting index for the next variant to avoid looping over all eqtls. 
							starting_index = i 
							setStartingIndex = False
					i += 1
					if i >= len(eqtl_positions):
						break

				rows.append([chrom, pos, n_eqtls])
	eqtl_counts_df = pd.DataFrame(rows, columns=['#Chrom', 'Pos', prefix + '_eqtls_' + str(bp) + 'bp'])
	return eqtl_counts_df


def main():
	#command line arguments

	cis_eqtl_dict = parse_eqtl_file('../../files/cis-eQTL_significant_20181017.txt')
	trans_eqtl_dict = parse_eqtl_file('../../files/trans-eQTL_significant_20181017.txt')
	print('eqtl files processed')

	data = pd.read_csv('../../data/data_noncoding.nctools-predictions.txt', sep='\t')
	print('data shape1', data.shape[0])
	
	data['#Chrom'] = data['#Chrom'].astype(str)
	data = data[['#Chrom', 'Pos']].drop_duplicates(subset=['#Chrom', 'Pos'])
	data = data.sort_values(by='#Chrom')

	print('data shape2', data.shape[0])

	eqtl_df_list = []

	for bp in [100, 250, 500, 1000, 2000, 5000, 10000]:
		df = count_eqtls(data, eqtl_dict=cis_eqtl_dict, prefix='cis', bp=bp)
		eqtl_df_list.append(df)

	for bp in [500, 1000, 2000, 5000, 10000, 20000]:
		df = count_eqtls(data, eqtl_dict=trans_eqtl_dict, prefix='trans', bp=bp)
		eqtl_df_list.append(df)

	for df in eqtl_df_list:
		print(df.shape)

	df_all_eqtl_counts = reduce(lambda x, y: pd.merge(x.reset_index(drop=True), y.reset_index(drop=True), on=['#Chrom', 'Pos'], how='inner'), eqtl_df_list)

	print(df_all_eqtl_counts.shape)
	# concatenated_dfs = pd.concat(eqtl_df_list)
	# print(concatenated_dfs.shape)
	df_all_eqtl_counts.to_csv('../../data/eqtls/eqtl_counts.txt', sep='\t', index=False)


if __name__ == '__main__':
	main()

