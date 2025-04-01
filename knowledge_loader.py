import os
import re
import json
import ollama
from pathlib import Path
from typing import List, Dict, Any

def process_file(file_path: str, modelname="nomic-embed-text") -> List[Dict[str, str]]:
    """
    Process a single text file into paragraphs with appropriate labels.
    
    Parameters:
    -----------
    file_path : str
        Path to the text file
    modelname : str
        Name of the model used for embedding the text
        
    Returns:
    --------
    List[Dict[str, str]]
        List of paragraph dictionaries with label, text, and embedding
    """
    try:
        # Get filename without extension for labeling
        file_name = os.path.basename(file_path)
        doc_name = os.path.splitext(file_name)[0]
        
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split content into paragraphs
        paragraphs = re.split(r'\n\s*\n', content)
        
        # Filter out empty paragraphs and strip whitespace
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Create the paragraph objects
        paragraph_objects = []
        for i, para_text in enumerate(paragraphs):
            paragraph_objects.append({
                "label": f"{doc_name}, paragraph_{i+1}",
                "text": para_text,
                "embedding": ollama.embeddings(model=modelname, prompt=para_text)["embedding"]
            })
        
        return paragraph_objects
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def save_to_json(paragraphs: List[Dict[str, str]], output_file: str):
    """
    Save paragraphs to a JSON file.
    
    Parameters:
    -----------
    paragraphs : List[Dict[str, str]]
        List of paragraph dictionaries to save
    output_file : str
        Path to the output JSON file
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(paragraphs, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(paragraphs)} paragraphs to {output_file}")
    except Exception as e:
        print(f"Error saving to {output_file}: {e}")

def load_docs(docs_dir: str = "./docs", output_dir: str = "./knowledge_base", modelname:str = "nomic-embed-text") -> Dict[str, List[Dict[str, str]]]:
    """
    Load all text documents from the docs directory and save JSON files to output directory.
    
    Parameters:
    -----------
    docs_dir : str
        Path to the directory containing document files
    output_dir : str
        Path to the directory where JSON files will be saved
        
    Returns:
    --------
    Dict[str, List[Dict[str, str]]]
        Dictionary with filenames as keys and lists of paragraph objects as values
    """
    result = {}
    
    # Ensure input docs directory exists
    if not os.path.exists(docs_dir):
        print(f"Directory not found: {docs_dir}")
        return result
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Get list of text files
    file_paths = [
        os.path.join(docs_dir, f) for f in os.listdir(docs_dir)
        if os.path.isfile(os.path.join(docs_dir, f)) and f.lower().endswith(('.txt', '.md'))
    ]
    
    if not file_paths:
        print(f"No text files found in {docs_dir}")
        return result
    
    print(f"Found {len(file_paths)} text files in {docs_dir}")
    
    # Process each file and save as individual JSON
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        doc_name = os.path.splitext(file_name)[0]
        
        # Process the file
        paragraphs = process_file(file_path, modelname)
        
        if paragraphs:
            result[doc_name] = paragraphs
            
            # Save individual JSON file with the same name as the original file
            json_path = os.path.join(output_dir, f"{doc_name}.json")
            save_to_json(paragraphs, json_path)
    
    return result

def save_combined_json(all_docs: Dict[str, List[Dict[str, str]]], output_dir: str = "./knowledge_base", output_file: str = "all_documents.json"):
    """
    Save all documents into a single combined JSON file.
    
    Parameters:
    -----------
    all_docs : Dict[str, List[Dict[str, str]]]
        Dictionary with all document paragraphs
    output_dir : str
        Directory to save the combined JSON file
    output_file : str
        Filename for the combined JSON file
    """
    # Flatten the dictionary into a single list of paragraphs
    all_paragraphs = []
    for doc_paragraphs in all_docs.values():
        all_paragraphs.extend(doc_paragraphs)
    
    if all_paragraphs:
        # Save to a single JSON file
        output_path = os.path.join(output_dir, output_file)
        save_to_json(all_paragraphs, output_path)
        return output_path
    
    return None

def main():
    """Main function to execute the knowledge loading process"""
    input_dir = "./docs"
    output_dir = "./knowledge_base"
    
    print(f"Loading documents from {input_dir}...")
    print(f"Saving JSON files to {output_dir}...")
    
    # Load and process all documents
    all_docs = load_docs(input_dir, output_dir, "nomic-embed-text")
    
    # Print summary
    total_paragraphs = sum(len(paragraphs) for paragraphs in all_docs.values())
    print(f"\nSummary:")
    print(f"- Processed {len(all_docs)} documents")
    print(f"- Extracted {total_paragraphs} total paragraphs")
    
    # Save combined JSON
    combined_file = save_combined_json(all_docs, output_dir)
    if combined_file:
        print(f"- All paragraphs saved to {combined_file}")

if __name__ == "__main__":
    main()