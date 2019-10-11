#!/bin/bash
#SBATCH --job-name=filter_Eigen_scores
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=15gb
#SBATCH --partition=short\
#SBATCH --output=/groups/umcg-gcc/tmp03/umcg-pfolkertsma/variant-prioritization/data/nctools_predictions/Eigen/filter_eigen_scores.out
#SBATCH --error=/groups/umcg-gcc/tmp03/umcg-pfolkertsma/variant-prioritization/data/nctools_predictions/Eigen/filter_eigen_scores.error


END=22
for ((i=1;i<=END;i++)); do
    echo "Processing chromosome "$i"..."
    awk 'NR==FNR {line[$1" "$2]; next} ($1" "$2 in line) {print $0; fflush();}' ../data/train_nc.vcf <(gzip -dc "../files/nctools_predictions/Eigen/Eigen_hg19_noncoding_annot_chr"$i".tab.gz") >> ../data/nctools_predictions/Eigen/eigen_filtered.txt
done
