import gzip
from collections import Counter

#only keeps the snvs from the train set (variants where ref allele and alt allele have length 1)
# train_snvs = [line.strip().split('\t') for line in open('../data/train_nc.vcf').readlines() if len(line.strip().split('\t')[3]) == 1 and len(line.strip().split('\t')[4]) == 1]

#only keep variants that share chromosome positions



# print(train_snvs[:10])


count = 0

train_snvs = {}
for line in open('../data/train_nc.vcf').readlines():
	line = line.strip().split('\t')
	count += 1
	if len(line[3]) == 1 and len(line[4]) == 1:
		x = (line[0], line[1])
		if x not in train_snvs:
			train_snvs[x] = []
		train_snvs[x].append([line[3], line[4]])
		# train_snvs.append(line)

print('All variants:', count)
print('Only SNVs:', len(train_snvs.keys()))
print(round(len(train_snvs.keys()) / count * 100, 2))

filtered = {}

for item in train_snvs:
	if len(train_snvs[item]) != 1:
		filtered[item] = train_snvs[item]

print('Percentage SNVs same chromosome position: ', round(len(filtered.keys()) / len(train_snvs.keys()) * 100, 2))

top_10 = sorted(filtered.keys(), key=lambda x: len(filtered[x]))[:10]

for p in top_10:
	print('Chr - Pos:', p)
	print(filtered[p])



# print(filtered[('20', '10621760')])

# print(train_snvs[1])
	
# 	s.add((line[0], line[1]))
# 	# print(line)
# 	# count += 1
# 	# if count == 10:
# 	# 	break


# count = 0
# with gzip.open("../files/nctools_predictions/ncboost_score_hg19_v20190108.tsv.gz", "rt") as f:
# 	for line in f:
# 		line = line.strip().split('\t')
# 		print(line)
# 		count += 1
# 		if count == 10:
# 			break


# print('COUNT')
# print(count)




# l = []

# for line in open('../data/train_nc.vcf').readlines():
# 	line = line.split('\t')
# 	if line[0] == '15' and line[1] == '35083508':
# 		print(line)
	# l.append((line[0], line[1]))

# print(Counter(l).keys())
# print(Counter(l).most_common(1))	[(('15', '35083508'), 17)]








