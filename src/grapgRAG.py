import os
import re
import json
import time
import glob
import logging
import argparse
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("GraphRAG")

try:
    import torch
    from torch import Tensor
    from transformers import AutoTokenizer, AutoModel
    from sentence_transformers import SentenceTransformer
    import networkx as nx
    from sklearn.metrics.pairwise import cosine_similarity
    import spacy
except ImportError as e:
    logger.error(f"Missing dependencies: {e}. Please install required packages.")
    logger.error("Run: pip install torch transformers sentence-transformers networkx scikit-learn spacy")
    logger.error("Also run: python -m spacy download en_core_web_sm")
    raise


class GraphRAGBuilder:
    """
    Building a knowledge graph enhanced retrieval augmented generation system.
    """

    def __init__(
        self,
        docs_dir: str = "./docs",
        output_dir: str = "./embedding_graph",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", 
        chunk_size: int = 256,
        overlap: int = 64,
        edge_threshold: float = 0.6
    ):
        """
        Initialize the GraphRAG system.
        
        Parameters:
        -----------
        docs_dir : str
            Directory containing source knowledge documents
        output_dir : str
            Directory to store the knowledge graph and embeddings
        embedding_model : str
            Model to use for generating embeddings
        chunk_size : int
            Maximum size of text chunks
        overlap : int
            Number of tokens to overlap between chunks
        edge_threshold : float
            Minimum similarity threshold for creating edges between nodes
        """
        self.docs_dir = Path(docs_dir)
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.edge_threshold = edge_threshold
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        
        # Initialize spaCy for entity extraction
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model for entity extraction")
        except OSError:
            logger.error("spaCy model not found. Downloading...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
    
    def load_documents(self) -> List[Dict[str, Any]]:
        """
        Load all text documents from the docs directory.
        
        Returns:
        --------
        List[Dict[str, Any]]:
            List of document dictionaries with metadata
        """
        documents = []
        
        for file_path in self.docs_dir.glob("**/*.txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                relative_path = file_path.relative_to(self.docs_dir)
                doc_id = str(relative_path).replace("\\", "/")
                
                document = {
                    "id": doc_id,
                    "content": content,
                    "title": file_path.stem,
                    "path": str(file_path),
                }
                documents.append(document)
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents from {self.docs_dir}")
        return documents
    
    def chunk_text(self, text: str, doc_id: str = "unknown") -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks of specified size.
        
        Parameters:
        -----------
        text : str
            Text to split into chunks
        doc_id : str
            Document identifier
            
        Returns:
        --------
        List[Dict[str, Any]]:
            List of chunk dictionaries with metadata
        """
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into sentences (simple approach)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size and we already have content
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append({
                    "id": f"{doc_id}_{chunk_id}",
                    "doc_id": doc_id,
                    "content": current_chunk.strip(),
                    "chunk_id": chunk_id
                })
                
                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_text = " ".join(words[-self.overlap:]) if len(words) > self.overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
                chunk_id += 1
            else:
                # Add sentence to current chunk
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append({
                "id": f"{doc_id}_{chunk_id}",
                "doc_id": doc_id,
                "content": current_chunk.strip(),
                "chunk_id": chunk_id
            })
        
        return chunks
    
    def extract_entities(self, text: str) -> List[str]:
        """
        Extract named entities from text using spaCy.
        
        Parameters:
        -----------
        text : str
            Text to extract entities from
            
        Returns:
        --------
        List[str]:
            List of extracted entity texts
        """
        doc = self.nlp(text[:10000])  # Limit size for spaCy processing
        
        # Extract entities of specific types (customize as needed)
        entity_types = ["PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "FAC"]
        entities = [ent.text for ent in doc.ents if ent.label_ in entity_types]
        
        # Also extract key noun phrases
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        
        # Combine and remove duplicates
        all_entities = list(set(entities + noun_phrases))
        
        return all_entities
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for text chunks.
        
        Parameters:
        -----------
        chunks : List[Dict[str, Any]]
            List of text chunks to embed
            
        Returns:
        --------
        List[Dict[str, Any]]:
            Chunks with added embeddings
        """
        texts = [chunk["content"] for chunk in chunks]
        
        # Generate embeddings in batches
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch_texts)
            all_embeddings.append(batch_embeddings)
            logger.info(f"Embedded batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        # Combine batches
        embeddings = np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i].tolist()
            
            # Extract entities
            chunk["entities"] = self.extract_entities(chunk["content"])
        
        return chunks
    
    def build_knowledge_graph(self, chunks: List[Dict[str, Any]]) -> nx.Graph:
        """
        Build a knowledge graph from text chunks using embeddings and entities.
        
        Parameters:
        -----------
        chunks : List[Dict[str, Any]]
            List of text chunks with embeddings and entities
            
        Returns:
        --------
        nx.Graph:
            NetworkX graph representing the knowledge
        """
        # Create graph
        G = nx.Graph()
        
        # Add nodes (chunks)
        for i, chunk in enumerate(chunks):
            G.add_node(
                chunk["id"],
                content=chunk["content"],
                doc_id=chunk["doc_id"],
                chunk_id=chunk["chunk_id"],
                entities=chunk["entities"]
            )
        
        # Create embedding matrix for similarity calculations
        embedding_matrix = np.array([chunk["embedding"] for chunk in chunks])
        
        # Add edges based on similarity and sequential relationships
        edges_added = 0
        
        # Group chunks by document
        doc_chunks = {}
        for chunk in chunks:
            doc_id = chunk["doc_id"]
            if doc_id not in doc_chunks:
                doc_chunks[doc_id] = []
            doc_chunks[doc_id].append(chunk)
        
        # Add sequential edges within documents
        for doc_id, doc_chunks_list in doc_chunks.items():
            # Sort by chunk_id
            doc_chunks_list.sort(key=lambda x: x["chunk_id"])
            
            # Connect sequential chunks
            for i in range(len(doc_chunks_list) - 1):
                chunk1 = doc_chunks_list[i]
                chunk2 = doc_chunks_list[i + 1]
                G.add_edge(
                    chunk1["id"],
                    chunk2["id"],
                    weight=1.0,
                    type="sequential"
                )
                edges_added += 1
        
        # Add semantic similarity edges
        similarity_matrix = cosine_similarity(embedding_matrix)
        
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                # Skip if already connected by sequential edge
                if G.has_edge(chunks[i]["id"], chunks[j]["id"]):
                    continue
                
                # Add edge if similarity is above threshold
                sim = similarity_matrix[i, j]
                if sim >= self.edge_threshold:
                    G.add_edge(
                        chunks[i]["id"],
                        chunks[j]["id"],
                        weight=float(sim),
                        type="semantic"
                    )
                    edges_added += 1
        
        # Add entity-based edges
        entity_to_chunks = {}
        for chunk in chunks:
            for entity in chunk["entities"]:
                if entity not in entity_to_chunks:
                    entity_to_chunks[entity] = []
                entity_to_chunks[entity].append(chunk["id"])
        
        # Connect chunks sharing entities
        for entity, chunk_ids in entity_to_chunks.items():
            if len(chunk_ids) > 1:
                for i in range(len(chunk_ids)):
                    for j in range(i + 1, len(chunk_ids)):
                        # Skip if already connected
                        if G.has_edge(chunk_ids[i], chunk_ids[j]):
                            continue
                        
                        G.add_edge(
                            chunk_ids[i],
                            chunk_ids[j],
                            weight=0.5,  # Lower weight for entity connections
                            type="entity",
                            entity=entity
                        )
                        edges_added += 1
        
        logger.info(f"Knowledge graph built with {len(G.nodes)} nodes and {len(G.edges)} edges")
        return G
    
    def save_knowledge_base(self, chunks: List[Dict[str, Any]], graph: nx.Graph) -> None:
        """
        Save the processed knowledge base to disk.
        
        Parameters:
        -----------
        chunks : List[Dict[str, Any]]
            List of processed text chunks
        graph : nx.Graph
            Knowledge graph
        """
        # Create metadata
        metadata = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "edge_threshold": self.edge_threshold,
            "chunks_count": len(chunks),
            "nodes_count": len(graph.nodes),
            "edges_count": len(graph.edges)
        }
        
        # Save metadata
        with open(self.output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        # Save chunks with embeddings
        chunks_file = self.output_dir / "chunks.json"
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)
        
        # Save graph in GML format (standard graph format)
        graph_file = self.output_dir / "knowledge_graph.gml"
        nx.write_gml(graph, graph_file)
        
        # Also save edges as JSON for easier processing
        edges = [{"source": u, "target": v, **d} for u, v, d in graph.edges(data=True)]
        edges_file = self.output_dir / "edges.json"
        with open(edges_file, "w", encoding="utf-8") as f:
            json.dump(edges, f, indent=2)
        
        logger.info(f"Knowledge base saved to {self.output_dir}")
    
    def process(self) -> None:
        """
        Process all documents to build and save the knowledge graph.
        """
        start_time = time.time()
        
        # Load documents
        documents = self.load_documents()
        
        # Process each document into chunks
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_text(doc["content"], doc["id"])
            all_chunks.extend(chunks)
        
        # Generate embeddings and extract entities
        logger.info(f"Generating embeddings for {len(all_chunks)} chunks")
        chunks_with_embeddings = self.generate_embeddings(all_chunks)
        
        # Build knowledge graph
        logger.info("Building knowledge graph")
        knowledge_graph = self.build_knowledge_graph(chunks_with_embeddings)
        
        # Save knowledge base
        self.save_knowledge_base(chunks_with_embeddings, knowledge_graph)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Processing completed in {elapsed_time:.2f} seconds")


def main():
    """Main function to run GraphRAG processing."""
    parser = argparse.ArgumentParser(description="Build a GraphRAG knowledge base from documents.")
    parser.add_argument("--docs_dir", type=str, default="./docs", help="Directory containing source documents")
    parser.add_argument("--output_dir", type=str, default="./embedding_graph", help="Output directory for knowledge base")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", 
                        help="Embedding model to use")
    parser.add_argument("--chunk_size", type=int, default=256, help="Maximum size of text chunks")
    parser.add_argument("--overlap", type=int, default=64, help="Overlap between chunks")
    parser.add_argument("--edge_threshold", type=float, default=0.6, 
                        help="Minimum similarity for creating edges")
    
    args = parser.parse_args()
    
    # Build knowledge base
    builder = GraphRAGBuilder(
        docs_dir=args.docs_dir,
        output_dir=args.output_dir,
        embedding_model=args.model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        edge_threshold=args.edge_threshold
    )
    
    builder.process()


if __name__ == "__main__":
    main()