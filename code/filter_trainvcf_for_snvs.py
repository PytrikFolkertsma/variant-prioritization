#filter train vcf file for snv's (needed for NCBoost and DeepSEA predictions)

vcf = open('../data/train_nc.vcf').readlines()
vcf_snvs = open('../data/train_nc_vcf_snvs.vcf', 'w')


vcf_snvs.write('chr\tstart\tend\tref\talt\trsid\n')

#chr	start		end		ref	alt	rsid

count = 0
for line in vcf:
	line = line.strip().split('\t')
	if len(line[3]) == 1 and len(line[4]) == 1:
		vcf_snvs.write('\t'.join([line[0], line[1], line[1], line[3], line[4], '']) + '\n')

vcf_snvs.close()