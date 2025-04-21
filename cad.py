import os
import faiss
import pickle
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
from PIL import Image
import matplotlib.pyplot as plt


np.set_printoptions(precision=3, suppress=True)

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 1: Load and clean captions
df = pd.read_csv('captions.txt', sep=",", names=["filename", "description"], skiprows=1, on_bad_lines='skip')
df.dropna(inplace=True)
filenames = df["filename"].tolist()
descriptions = df["description"].tolist()

# Step 2: Encode the descriptions into embeddings
embeddings = model.encode(descriptions, convert_to_numpy=True)

# Print first 5 embedding vectors
print("\nFirst 5 embedding vectors:")
for i in range(5):
    print(f"{i+1}: {embeddings[i]}")

# Step 3: Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Step 4: Save FAISS index and metadata
faiss.write_index(index, "image_index.faiss")
with open("metadata.pkl", "wb") as f:
    pickle.dump((filenames, descriptions), f)

print(f"\nIndexed {len(filenames)} image captions successfully.")

# Step 5: Search function
def search_images(query, top_k=3):
    with open("metadata.pkl", "rb") as f:
        filenames, descriptions = pickle.load(f)
    index = faiss.read_index("image_index.faiss")

    # Encode and normalize query
    query_embedding = model.encode([query], convert_to_numpy=True)
    query_embedding = query_embedding / norm(query_embedding)

    # Search more than top_k to filter duplicates
    raw_scores, raw_indices = index.search(query_embedding, top_k * 5)

    unique_results = {}
    for idx in raw_indices[0]:
        file = filenames[idx]
        if file not in unique_results:
            caption = descriptions[idx]
            caption_embedding = model.encode([caption], convert_to_numpy=True)[0]
            cosine_sim = np.dot(query_embedding[0], caption_embedding) / (
                norm(query_embedding[0]) * norm(caption_embedding)
            )
            unique_results[file] = (caption, cosine_sim)
        if len(unique_results) == top_k:
            break

    # Sort results by cosine similarity
    results = [(os.path.join("images", file), cap, score) for file, (cap, score) in unique_results.items()]
    results.sort(key=lambda x: x[2], reverse=True)
    return results

# Step 6: Run the search interface
if _name_ == "_main_":
    while True:
        query = input("\nEnter your search prompt (or 'exit'): ")
        if query.lower() == 'exit':
            break
        results = search_images(query)
        print("\nTop 3 matching captions with cosine similarity:")
        for path, desc, score in results:
            print(f"Image: {path}, Caption: {desc}, Cosine Similarity: {score:.4f}")
            try:
                img = Image.open(path)
                plt.figure(figsize=(3, 3))
                plt.imshow(img)
                plt.axis("off")
                plt.title(f"Score: {score:.2f}")
                plt.show()
            except Exception as e:
                print(f"Could not load image: {path}. Error: {e}")