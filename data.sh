mkdir -p hg38/
curl https://storage.googleapis.com/basenji_barnyard2/hg38.ml.fa.gz > hg38/hg38.ml.fa.gz
ungzip hg38/hg38.ml.fa.gz
curl https://storage.googleapis.com/basenji_barnyard2/sequences_human.bed > hg38/human-sequences.bed

