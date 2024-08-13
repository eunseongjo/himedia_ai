# STEP 1
from sentence_transformers import SentenceTransformer

# STEP 2
# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# STEP 3
# The sentences to encode
sentence1 = "나는 오늘 점심에 간짜장을 먹었어 너무 맛있었어 내일은 굶을려고."
sentence2 = "im hungry. i like 간짜장."

# STEP 4
# 2. Calculate embeddings by calling model.encode()
embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)

print(embedding1.shape)
print(embedding2.shape)
# [3, 384]

# STEP 5
# 3. Calculate the embedding similarities
similarities = model.similarity(embedding1, embedding2)
print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])