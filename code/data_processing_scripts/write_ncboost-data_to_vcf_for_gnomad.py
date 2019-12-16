
ncboost_vcf = open('../data/ncboost_pathogenic_variants.vcf', 'w')

for line in open('../files/NCBoost_pathogenic_variants/13059_2019_1634_MOESM1_ESM.txt').readlines():
	line = line.split('\t')
	ncboost_vcf.write('\t'.join([line[0], line[1], '.', line[2], line[3]]) + '\n')

ncboost_vcf.close()