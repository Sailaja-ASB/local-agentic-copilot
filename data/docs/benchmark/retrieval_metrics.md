# Retrieval Evaluation Metrics

Top-1 accuracy is the share of questions where the correct source is ranked first. Hit@k, also called Recall@k when there is one relevant document, is the share where the correct source appears anywhere in the top k results. Mean Reciprocal Rank (MRR) averages 1 / rank of the first correct result, so rank 1 scores 1.0, rank 2 scores 0.5 and rank 3 scores 0.33.

A fixed benchmark of questions with known expected sources, written before looking at results, allows fair comparison between retrieval methods. Small benchmarks give noisy numbers: with 10 questions, one change moves accuracy by 10 points.
