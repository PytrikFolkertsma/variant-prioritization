

annovar = open('../data/data_nc.ncboost.variant_function')
cadd = open('../data/data_nc.ncboost_cadd.tsv')

data = open('../data/data_nc.txt')

# for line in data:
# 	line = line.strip().split('\t')
# 	if line 
	#filter for snvs

#for each line in annovar:
	#extract corect CADD features from data.
	#write fo file

count = 0

for line in annovar:
	line = line.strip().split('\t')
	print(line)
	count += 1
	if count == 10:
		break