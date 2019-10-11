import sys
import gzip
count = 0
with gzip.open(sys.argv[1], "rt") as f:
	for line in f:
		line = line.strip().split('\t')
		print(line)
		count += 1
		if count == 10:
			break