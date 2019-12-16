#!/bin/bash
#SBATCH --job-name=ncboost_CADD
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=40GB
#SBATCH --partition=short

ml CADD/v1.4 
CADD.sh -a -g GRCh37 -o /groups/umcg-gcc/tmp04/umcg-pfolkertsma/variant-prioritization/data/ncboost_pathogenic-variants_CADD-annotated.tsv.gz /groups/umcg-gcc/tmp04/umcg-pfolkertsma/variant-prioritization/data/ncboost_pathogenic_variants.vcf
