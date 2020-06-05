# Speakers coreference resolution

please setup the environment as follows:

1. Environmental requirements: requirements.txt 
2. Download pretrained Chinese embeddings from: https://github.com/Embedding/Chinese-Word-Vectors, select Zhihu_QA Corpus, context features: word.
3. Setup the ELMo module using codes provided by: https://github.com/HIT-SCIR/ELMoForManyLangs, select the simplified-Chinese ELMo.
4. Filter pretrained embeddings: filter_embeddings.py YourWordEmbeddingPath data/train.law.jsonlines data/dev.law.jsonlines data/test.law.jsonlines
5. Train your model with: python Train.py YourSettingName
6. Evaluate your model with: python Evaluate.py YourSettingName

# Acknowledgment
We built the training framework based on the original [Incorporating Context and External Knowledge for Pronoun Coreference Resolution](https://www.aclweb.org/anthology/N19-1093.pdf/)

Pretrained Embeddings are provided by:
[Analogical Reasoning on Chinese Morphological and Semantic Relations](https://github.com/Embedding/Chinese-Word-Vectors)

Pretrained ELMo Embeddings are based on: [Word vectors, reuse, and replicability: Towards a community repository of large-text resources](https://www.aclweb.org/anthology/W17-0237.pdf)