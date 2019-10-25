
import pyBigWig
import pandas as pd

bw = pyBigWig.open("../files/nctools_predictions/LINSIGHT.bw")

newFile = open('../data/nctools_predictions/linsight_predictions.txt', 'w')

count = 0
for line in open('../data/data_nc.vcf'):
	line = line.strip().split('\t')
	count += 1
	if count == 1:
		newFile.write('\t'.join([line[0], line[1], line[2], line[3], 'LINSIGHT_score']) + '\n')
		continue
	if line[0] != 'Y':
		if len(line[2]) == 1 and len(line[3]) == 1:
			value = bw.values('chr' + line[0], int(line[1])-1, int(line[1]))
			newFile.write('\t'.join([line[0], line[1], line[2], line[3], str(value[0])]) + '\n')
	if count % 1000 == 0:
		print(count, 'lines processed')

newFile.close()
