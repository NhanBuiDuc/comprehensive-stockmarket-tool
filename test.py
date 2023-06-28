from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained SentenceTransformer model
model = SentenceTransformer('ProsusAI/finbert')

# Define the sentences
sentence1 = "Many investors seeing substantial gains in their portfolios."
sentence2 = "A surge in market activity has resulted in numerous investors witnessing remarkable growth in the value of their investment portfolios."
sentence3 = "She took a deep breath to speak in front of the audience."

# Get the embeddings for the sentences
embeddings = model.encode([sentence1, sentence2, sentence3])

# Print the embeddings
print("Embeddings:")
for i, embedding in enumerate(embeddings):
    print(f"Sentence {i+1}: {embedding}")

# Calculate the cosine similarity
similarity = cosine_similarity(embeddings)

# Print the similarity matrix
print("\nSimilarity Matrix:")
for i in range(len(similarity)):
    for j in range(len(similarity[i])):
        print(f"The cosine similarity between sentence {i+1} and sentence {j+1} is: {similarity[i][j]}")
