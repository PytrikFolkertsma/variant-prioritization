newFile = open('../../NCBoost/output/data_nc.ncboost.vcf', 'w')
for line in open('../data/data_nc.vcf').readlines()[1:]:
	line = line.strip().split('\t')
	if len(line[2]) == 1 and len(line[3]) == 1:
		newFile.write('\t'.join([line[0], line[1], line[1], line[2], line[3]]) + '\n')
