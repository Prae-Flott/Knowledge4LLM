import os
import json
import numpy as np
from numpy.linalg import norm
import ollama
from typing import List, Dict, Any, Tuple

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

# New functions for knowledge base

def load_knowledge_embeddings(knowledge_dir: str = "./knowledge_base") -> List[Dict[str, Any]]:
    """
    Load all knowledge items with embeddings from the knowledge base directory.
    
    Parameters:
    -----------
    knowledge_dir : str
        Path to the knowledge base directory containing JSON files
        
    Returns:
    --------
    List[Dict[str, Any]]
        List of knowledge items with labels, text, and embeddings
    """
    knowledge_items = []
    
    # Check if directory exists
    if not os.path.exists(knowledge_dir):
        print(f"Knowledge base directory not found: {knowledge_dir}")
        return knowledge_items
    
    # Find all JSON files in the directory
    json_files = [f for f in os.listdir(knowledge_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in {knowledge_dir}")
        return knowledge_items
    
    # Load each JSON file
    for json_file in json_files:
        file_path = os.path.join(knowledge_dir, json_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            # Check if content is a list or single item
            if isinstance(content, list):
                # Process each item in the list
                for item in content:
                    if "label" in item and "text" in item and "embedding" in item:
                        knowledge_items.append(item)
            elif isinstance(content, dict) and "label" in content and "text" in content and "embedding" in content:
                # Process a single item
                knowledge_items.append(content)
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    print(f"Loaded {len(knowledge_items)} items with embeddings from knowledge base")
    return knowledge_items

def extract_embeddings_from_knowledge(knowledge_items: List[Dict[str, Any]]) -> List[List[float]]:
    """
    Extract just the embedding vectors from knowledge items.
    
    Parameters:
    -----------
    knowledge_items : List[Dict[str, Any]]
        List of knowledge items with embeddings
        
    Returns:
    --------
    List[List[float]]
        List of embedding vectors
    """
    return [item["embedding"] for item in knowledge_items if "embedding" in item]

def find_similar_knowledge(prompt: str, knowledge_items: List[Dict[str, Any]], 
                          modelname: str = "nomic-embed-text", top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Find knowledge items most similar to the prompt.
    
    Parameters:
    -----------
    prompt : str
        The search prompt
    knowledge_items : List[Dict[str, Any]]
        List of knowledge items with embeddings
    modelname : str
        Name of the embedding model
    top_k : int
        Number of top similar items to return
        
    Returns:
    --------
    List[Dict[str, Any]]
        List of most similar knowledge items with similarity scores
    """
    if not knowledge_items:
        print("No knowledge items available")
        return []
    
    # Extract embeddings from knowledge items
    embeddings = extract_embeddings_from_knowledge(knowledge_items)
    
    if not embeddings:
        print("No embeddings found in knowledge items")
        return []
    
    # Generate embedding for the prompt
    try:
        prompt_embedding = ollama.embeddings(model=modelname, prompt=prompt)["embedding"]
        
        # Find most similar items
        similarities = find_most_similar(prompt_embedding, embeddings)
        
        # Get top_k results
        top_results = []
        for similarity, idx in similarities[:top_k]:
            # Create a result object with original item plus similarity score
            result = {
                "label": knowledge_items[idx]["label"],
                "text": knowledge_items[idx]["text"],
                "similarity": similarity
            }
            top_results.append(result)
            
        return top_results
        
    except Exception as e:
        print(f"Error finding similar knowledge: {e}")
        return []

def get_knowledge_context(prompt: str, knowledge_dir: str = "./knowledge_base", 
                        modelname: str = "nomic-embed-text", top_k: int = 5) -> Tuple[str, List[str]]:
    """
    Get relevant context from knowledge base for a prompt.
    
    Parameters:
    -----------
    prompt : str
        The search prompt
    knowledge_dir : str
        Path to the knowledge base directory
    modelname : str
        Name of the embedding model
    top_k : int
        Number of top similar items to include
        
    Returns:
    --------
    Tuple[str, List[str]]
        Combined context text and list of source labels
    """
    # Load knowledge items
    knowledge_items = load_knowledge_embeddings(knowledge_dir)
    
    if not knowledge_items:
        return "", []
    
    # Find similar items
    similar_items = find_similar_knowledge(prompt, knowledge_items, modelname, top_k)
    
    if not similar_items:
        return "", []
    
    # Extract text and labels
    context_text = "\n\n".join(item["text"] for item in similar_items)
    source_labels = [item["label"] for item in similar_items]
    
    return context_text, source_labels

# Example usage
if __name__ == "__main__":
    # Simple test of the new functions
    prompt = input("Enter your question: ")
    context, sources = get_knowledge_context(prompt)
    
    if sources:
        print("\nRelevant sources:")
        for src in sources:
            print(f"- {src}")
    
    if context:
        print("\nContext for LLM:")
        print(context[:200] + "..." if len(context) > 200 else context)
    else:
        print("No relevant context found")