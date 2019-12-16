newFile = open('../data/NCBoost_gnomdAD_annotated.txt', 'w')
newFile.write('#Chrom\tPos\tRef\tAlt\tmax_AF\n')
for line in open('../data/NCBoost_annotated_gnomAD.txt'):
	line = line.strip().split('\t')
	if len(line) == 8 and line[7] != '.':
		info = line[7].split(';')
		af = ''
		for item in info:
			if 'gnomad_AF_POPMAX' in item:
				af = item
			elif 'exomes_AF_POPMAX' in item:
				af = item
			else:
				continue
		newFile.write('\t'.join([line[0], line[1], line[3], line[4], af.split('=')[1]]) + '\n')
	elif len(line) == 8 and line[7] == '.':
		newFile.write('\t'.join([line[0], line[1], line[3], line[4], '0.0']) + '\n')

newFile.close()