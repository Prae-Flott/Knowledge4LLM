import os
import re
import json
import ollama
import time
import networkx as nx
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Import project modules
from utils.text_spliter import parse_file
from data_loader import read_latest_description
from utils.evaluation_utils import (
    load_test_questions,
    normalize_text,
    calculate_similarity,
    calculate_token_metrics,
    save_results,
    print_metrics_summary
)
from sentence_transformers import SentenceTransformer, util

def count_tokens(text: str) -> int:
    """
    Simple approximation of token count
    """
    return len(text.split())

def load_knowledge_graph(kb_dir: str = "./embedding_graph") -> Tuple[List[Dict], nx.Graph, Dict]:
    """
    Load the knowledge graph and chunked embeddings.
    
    Parameters:
    -----------
    kb_dir : str
        Path to the knowledge graph directory
        
    Returns:
    --------
    Tuple[List[Dict], nx.Graph, Dict]:
        Chunks with embeddings, knowledge graph, and metadata
    """
    # Load chunks with embeddings
    chunks_file = os.path.join(kb_dir, "chunks.json")
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    # Load graph
    graph_file = os.path.join(kb_dir, "knowledge_graph.gml")
    graph = nx.read_gml(graph_file)
    
    # Load metadata
    metadata_file = os.path.join(kb_dir, "metadata.json")
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    print(f"Loaded knowledge graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    return chunks, graph, metadata

def get_graph_context(
        query: str, 
        chunks: List[Dict], 
        graph: nx.Graph,
        model: SentenceTransformer,
        top_k: int = 10,
        expansion_hops: int = 1
    ) -> Tuple[str, List[str]]:
    """
    Retrieve relevant context using graph-based retrieval.
    
    Parameters:
    -----------
    query : str
        The query to retrieve context for
    chunks : List[Dict]
        List of chunks with embeddings
    graph : nx.Graph
        Knowledge graph
    model : SentenceTransformer
        Embedding model
    top_k : int
        Number of initial chunks to retrieve
    expansion_hops : int
        Number of hops to expand in the graph
        
    Returns:
    --------
    Tuple[str, List[str]]:
        Combined context string and list of source chunk IDs
    """
    # Encode query
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Get embeddings from chunks
    chunk_embeddings = np.array([chunk["embedding"] for chunk in chunks])
    
    # Calculate similarities with query
    similarities = util.pytorch_cos_sim(query_embedding, chunk_embeddings)[0].cpu().numpy()
    
    # Get top-k chunk indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Get initial chunk IDs
    initial_chunk_ids = [chunks[idx]["id"] for idx in top_indices]
    
    # Expand through graph
    expanded_chunk_ids = set(initial_chunk_ids)
    current_set = set(initial_chunk_ids)
    
    # Perform BFS-like expansion through the graph
    for _ in range(expansion_hops):
        next_set = set()
        for chunk_id in current_set:
            if chunk_id in graph:
                # Add neighbors
                neighbors = list(graph.neighbors(chunk_id))
                
                # Filter to highest weighted edges if we have too many
                if len(neighbors) > 3:
                    # Get edge weights for all neighbors
                    weights = [graph[chunk_id][neighbor]['weight'] for neighbor in neighbors]
                    # Sort by weight
                    sorted_neighbors = [n for _, n in sorted(zip(weights, neighbors), reverse=True)]
                    neighbors = sorted_neighbors[:3]  # Take top 3
                
                next_set.update(neighbors)
        
        # Add new neighbors to expanded set
        expanded_chunk_ids.update(next_set)
        current_set = next_set
    
    # Get unique expanded chunks
    all_chunk_ids = list(expanded_chunk_ids)
    
    # Re-rank expanded chunks by similarity to query
    chunk_id_to_idx = {chunk["id"]: i for i, chunk in enumerate(chunks)}
    expanded_indices = [chunk_id_to_idx[cid] for cid in all_chunk_ids if cid in chunk_id_to_idx]
    expanded_similarities = similarities[expanded_indices]
    
    # Sort by similarity
    sorted_indices = np.argsort(expanded_similarities)[::-1]
    sorted_chunk_ids = [all_chunk_ids[i] for i in sorted_indices]
    
    # Get content for sorted chunks
    chunk_id_to_content = {chunk["id"]: chunk["content"] for chunk in chunks}
    context_parts = []
    retrieved_ids = []
    
    # Filter to top chunks (based on expanded and re-ranked)
    for chunk_id in sorted_chunk_ids[:15]:  # Limit to 15 chunks
        if chunk_id in chunk_id_to_content:
            context_parts.append(chunk_id_to_content[chunk_id])
            retrieved_ids.append(chunk_id)
    
    # Combine context parts
    context = "\n\n".join(context_parts)
    
    return context, retrieved_ids

def main(include_data: bool = False, use_graph: bool = True) -> None:
    """
    Main evaluation function using graph-based retrieval.
    
    Parameters:
    -----------
    include_data : bool
        Whether to include latest battery data
    use_graph : bool
        Whether to use graph-based retrieval (vs. simple embedding)
    """
    # Configuration
    model_name = "deepseek-r1:1.5b"
    test_file = "test_QA"
    graph_kb_dir = "./embedding_graph"
    traditional_kb_dir = "./knowledge_base"
    top_k = 10
    output_dir = "./evaluation_results"
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    retrieval_type = "graph" if use_graph else "vector"
    results_file = f"{output_dir}/eval_{retrieval_type}_{model_name.replace(':', '_')}_{timestamp}.json"
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    SYSTEM_PROMPT = (
        "You are a helpful reading assistant who answers questions based on snippets of text provided in context. "
        "Answer only using the context provided, being as concise as possible. If you're unsure, just say that you don't know.\n"
        "Instructions: Give a brief, to-the-point answer. Keep your answer as short as possible.\n"
        "Context:\n\n"
    )

    # Add latest battery data if requested
    if include_data:
        battery_data = read_latest_description()
        if battery_data:
            results_file = f"{output_dir}/{retrieval_type}_eval_with_data_{model_name.replace(':', '_')}_{timestamp}.json"
            print(f"Including latest battery data in evaluation")

    # Load test questions
    q_a_pairs = load_test_questions(test_file, docs_dir="./docs")
    if not q_a_pairs:
        print("No test questions found. Exiting.")
        return
    
    # Load embedding model
    print(f"Loading embedding model: {embedding_model_name}")
    embedding_model = SentenceTransformer(embedding_model_name)
    
    # Load knowledge graph or traditional embeddings based on mode
    if use_graph:
        print("Using graph-based retrieval")
        chunks, graph, metadata = load_knowledge_graph(graph_kb_dir)
    
    print(f"Starting evaluation with {len(q_a_pairs)} questions using model {model_name}...")
    
    # Initialize results storage
    results = {
        "model": model_name,
        "test_file": test_file,
        "timestamp": timestamp,
        "retrieval_type": retrieval_type,
        "questions": [],
        "metrics": {}
    }
    
    # Track metrics
    correct_count = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_similarity = 0
    total_time = 0
    total_tokens = 0
    total_context_quality = 0  # Track quality of retrieved context
    total_retrieval_time = 0
    
    # Process each question-answer pair
    for i, pair in enumerate(q_a_pairs):
        user_query = pair["question"]
        
        # Add data to the query if requested
        if include_data and battery_data:
            user_query = f"Given this battery information: '{battery_data}', please answer: {user_query}"
        
        expected_answer = pair["answer"]
        
        print(f"\n[{i+1}/{len(q_a_pairs)}] Q: {user_query}")
        print(f"Expected: {expected_answer}")

        # Start retrieval timing
        retrieval_start = time.time()
        
        # Retrieve relevant context - either graph-based or traditional
        if use_graph:
            context, sources = get_graph_context(
                user_query, 
                chunks, 
                graph,
                embedding_model,
                top_k=top_k,
                expansion_hops=1
            )
        else:
            # Fall back to traditional retrieval
            from utils.embedding_utils import get_knowledge_context
            context, sources = get_knowledge_context(
                user_query, 
                knowledge_dir=traditional_kb_dir, 
                modelname="nomic-embed-text", 
                top_k=top_k
            )
        
        # End retrieval timing
        retrieval_time = time.time() - retrieval_start
        total_retrieval_time += retrieval_time
        
        # Start inference timing
        start_time = time.time()
        
        # Create a chat prompt by combining a system prompt and the context
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT + context,
                },
                {"role": "user", "content": user_query},   
            ],
        )
        
        # End timing
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Get the actual answer
        model_answer = response["message"]["content"].strip()
        
        # Remove thinking part from the answer
        if "<think>\n" in model_answer and "</think>\n" in model_answer:    
            actual_answer = re.sub(r'<think>\n.*?</think>\n\n', '', model_answer, flags=re.DOTALL)
        else:
            actual_answer = model_answer
        print(f"Model: {actual_answer}")
        
        # Count tokens in the response
        token_count = count_tokens(actual_answer)
        tokens_per_second = token_count / generation_time if generation_time > 0 else 0
        time_per_token = generation_time / token_count if token_count > 0 else 0
        
        print(f"Retrieval time: {retrieval_time:.2f}s")
        print(f"Generation time: {generation_time:.2f}s for {token_count} tokens")
        print(f"Time per token: {time_per_token*1000:.2f} ms")
        print(f"Tokens per second: {tokens_per_second:.2f}")

        # Calculate metrics
        token_metrics = calculate_token_metrics(expected_answer, actual_answer)
        
        # Calculate context quality: similarity between expected answer and context
        context_similarity = calculate_similarity(expected_answer, context)
        total_context_quality += context_similarity
        
        # Determine if recall meets threshold
        if token_metrics['recall'] >= 0.9:
            correct_count += 1
        
        # Update totals
        total_precision += token_metrics['precision']
        total_recall += token_metrics['recall']
        total_f1 += token_metrics['f1']
        total_time += generation_time
        total_tokens += token_count
        
        # Print metric summary
        print(f"Metrics:")
        print(f"  - Precision: {token_metrics['precision']:.2f}")
        print(f"  - Recall: {token_metrics['recall']:.2f}")
        print(f"  - F1: {token_metrics['f1']:.2f}")
        print(f"  - Context quality: {context_similarity:.2f}")
        
        # Store result
        question_result = {
            "question": user_query,
            "expected_answer": expected_answer,
            "actual_answer": model_answer,
            "precision": token_metrics['precision'],
            "recall": token_metrics['recall'],
            "f1": token_metrics['f1'],
            "sources": sources[:5],  # Store top 5 sources
            "context_quality": context_similarity,
            "generation_time": generation_time,
            "retrieval_time": retrieval_time,
            "token_count": token_count,
            "time_per_token": time_per_token,
            "tokens_per_second": tokens_per_second
        }
        results["questions"].append(question_result)
    
    # Calculate averages
    total_questions = len(q_a_pairs)
    avg_time_per_token = total_time / total_tokens if total_tokens > 0 else 0
    avg_tokens_per_second = total_tokens / total_time if total_time > 0 else 0
    avg_context_quality = total_context_quality / total_questions if total_questions > 0 else 0
    avg_retrieval_time = total_retrieval_time / total_questions if total_questions > 0 else 0
    
    # Calculate overall metrics
    results["metrics"] = {
        "total_questions": total_questions,
        "correct_answers": correct_count,
        "accuracy": correct_count / total_questions if total_questions > 0 else 0,
        "precision": total_precision / total_questions if total_questions > 0 else 0,
        "recall": total_recall / total_questions if total_questions > 0 else 0,
        "f1": total_f1 / total_questions if total_questions > 0 else 0,
        "context_quality": avg_context_quality,
        "retrieval_time": avg_retrieval_time,
        "total_generation_time": total_time,
        "total_tokens_generated": total_tokens,
        "avg_time_per_token": avg_time_per_token,
        "avg_time_per_token_ms": avg_time_per_token * 1000,  # in milliseconds
        "avg_tokens_per_second": avg_tokens_per_second
    }
    
    # Print overall results
    print("\n" + "="*50)
    print(f"EVALUATION SUMMARY ({retrieval_type.upper()} RETRIEVAL)")
    print("="*50)
    print(f"Total questions: {total_questions}")
    print(f"Correct answers: {correct_count}")
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"Precision: {results['metrics']['precision']:.4f}")
    print(f"Recall: {results['metrics']['recall']:.4f}")
    print(f"F1 Score: {results['metrics']['f1']:.4f}")
    print(f"Context Quality: {results['metrics']['context_quality']:.4f}")
    print(f"\nRetrieval Performance:")
    print(f"  - Avg. retrieval time: {avg_retrieval_time:.2f}s")
    print(f"\nGeneration Performance:")
    print(f"  - Total time: {total_time:.2f}s")
    print(f"  - Total tokens: {total_tokens}")
    print(f"  - Avg. time per token: {avg_time_per_token*1000:.2f} ms")
    print(f"  - Avg. tokens per second: {avg_tokens_per_second:.2f}")
    
    # Save results to file
    save_results(results, results_file)
    print(f"\nFull evaluation results saved to {results_file}")


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate LLM performance with graph-based retrieval')
    parser.add_argument('--data', action='store_true', help='Include latest battery data in the prompt')
    parser.add_argument('--traditional', action='store_true', help='Use traditional vector retrieval instead of graph')
    
    args = parser.parse_args()
    
    # Run evaluation with command line arguments
    main(include_data=args.data, use_graph=not args.traditional)