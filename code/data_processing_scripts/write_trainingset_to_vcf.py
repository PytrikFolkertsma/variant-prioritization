

trainset = open('../data/train_nc.txt').readlines()
trainset_vcf = open('../data/train_nc.vcf', 'w')

firstline = trainset[0].split('\t')
f = lambda x : firstline.index(x)

indices = [f('#Chrom'), f('Pos'), f('Ref'), f('Alt')]

for line in trainset[1:]:
	line = line.split('\t')
	if not (len(line[indices[3]]) > 100 or len(line[indices[2]]) > 100):
		trainset_vcf.write('\t'.join([line[indices[0]], line[indices[1]], '', line[indices[2]], line[indices[3]]]) + '\n')

trainset_vcf.close()