#!/bin/bash
#SBATCH --job-name=filter_Eigen_scores
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=20gb
#SBATCH --partition=short\
#SBATCH --output=/groups/umcg-gcc/tmp03/umcg-pfolkertsma/variant-prioritization/data/nctools_predictions/filter_ReMM_scores.out.txt
#SBATCH --error=/groups/umcg-gcc/tmp03/umcg-pfolkertsma/variant-prioritization/data/nctools_predictions/filter_ReMM_scores.error.txt

awk 'NR==FNR {line[$1" "$2]; next} ($1" "$2 in line) {print $0; fflush();}' ../data/data_nc.vcf <(gzip -dc "../files/nctools_predictions/ReMM/ReMM.v0.3.1.tsv.gz") >> ../data/nctools_predictions/ReMM_filtered.txt
