import os
import json
import numpy as np
from numpy.linalg import norm
import ollama

def save_embeddings(filename, embeddings):
    """
    Save embeddings to a JSON file in the 'embeddings' directory.
    """
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    with open(f"embeddings/{filename}.json", "w") as f:
        json.dump(embeddings, f)

def load_embeddings(filename):
    """
    Load embeddings from a JSON file if it exists.
    """
    path = f"embeddings/{filename}.json"
    if not os.path.exists(path):
        return False
    with open(path, "r") as f:
        return json.load(f)

def get_embeddings(filename, modelname, chunks):
    """
    Get embeddings for each text chunk using an Ollama model.
    If the embeddings have been saved already, load them; otherwise, generate and save them.
    """
    if (embeddings := load_embeddings(filename)) is not False:
        return embeddings
    embeddings = [
        ollama.embeddings(model=modelname, prompt=chunk)["embedding"]
        for chunk in chunks
    ]
    save_embeddings(filename, embeddings)
    return embeddings

def find_most_similar(needle, haystack):
    """
    Compute the cosine similarity between a given 'needle' embedding and a list of 'haystack' embeddings.
    Returns a sorted list of tuples (similarity_score, index) in descending order.
    """
    needle_norm = norm(needle)
    similarity_scores = [
        np.dot(needle, item) / (needle_norm * norm(item))
        for item in haystack
    ]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)
