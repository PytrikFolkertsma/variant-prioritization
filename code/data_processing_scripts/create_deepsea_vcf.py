newFile = open('../data/data_nc.deepsea.vcf', 'w')
for line in open('../data/data_nc.vcf').readlines()[1:]:
	line = line.strip().split('\t')
	if len(line[2]) < 100 and len(line[3]) < 100:
		newFile.write('\t'.join([line[0], line[1], '', line[2], line[3]]) + '\n')
newFile.close()