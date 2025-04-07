import os
import re
import json
import ollama
from datetime import datetime
from typing import Dict, List, Any

# Import project modules
from utils.text_spliter import parse_file
from data_loader import read_latest_description
from utils.embedding_utils import get_knowledge_context
from utils.evaluation_utils import (
    load_test_questions,
    normalize_text,
    calculate_similarity,
    calculate_token_metrics,
    is_correct_answer,
    save_results,
    print_metrics_summary
)

def main(include_data=False):
    """
    Main evaluation function that processes test questions and calculates metrics.
    """
    # Configuration
    model_name = "deepseek-r1:1.5b"
    test_file = "test_QA"
    knowledge_dir = "./knowledge_base"
    top_k = 10
    output_dir = "./evaluation_results"
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{output_dir}/eval_{model_name.replace(':', '_')}_{timestamp}.json"
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    SYSTEM_PROMPT = (
        "You are a helpful reading assistant who answers questions based on snippets of text provided in context. "
        "Answer only using the context provided, being as concise as possible. If you're unsure, just say that you don't know.\n"
        "Instructions: AnGive a brief, to-the-point answer. Keep your answer as short as possible.\n"
        "Context:\n\n"
    )

     # Add latest battery data if requested
    if include_data:
        battery_data = read_latest_description()
        if battery_data:
            results_file = f"{output_dir}/eval_with_data_{model_name.replace(':', '_')}_{timestamp}.json"
            print(f"Including latest battery data in evaluation")

    # Load test questions
    q_a_pairs = load_test_questions(test_file, docs_dir="./docs")
    if not q_a_pairs:
        print("No test questions found. Exiting.")
        return
    
    print(f"Starting evaluation with {len(q_a_pairs)} questions using model {model_name}...")
    
    # Initialize results storage
    results = {
        "model": model_name,
        "test_file": test_file,
        "timestamp": timestamp,
        "questions": [],
        "metrics": {}
    }
    
    # Track correct answers and metrics
    correct_count = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_similarity = 0
    
    # Process each question-answer pair
    for i, pair in enumerate(q_a_pairs):
        
        user_query = pair["question"]
        
        # add the data to the query if requested
        if include_data and battery_data:
            user_query = f"Given this battery information: '{battery_data}', please answer: {user_query}"
        
        expected_answer = pair["answer"]
        
        print(f"\n[{i+1}/{len(q_a_pairs)}] Q: {user_query}")
        print(f"Expected: {expected_answer}")

        # Retrieve relevant context from knowledge base
        context, sources = get_knowledge_context(
            user_query, 
            knowledge_dir=knowledge_dir, 
            modelname="nomic-embed-text", 
            top_k=top_k)

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
        
        # Get the actual answer
        model_answer = response["message"]["content"].strip()
        
        # remove the thiking part from the answer
        if "<think>\n" in model_answer and "</think>\n" in model_answer:    
            actual_answer = re.sub(r'<think>\n.*?</think>\n\n', '', model_answer, flags=re.DOTALL)
        else:
            actual_answer = model_answer
        print(f"Model: {actual_answer}")

        # Calculate metrics
        token_metrics = calculate_token_metrics(expected_answer, actual_answer)
        
        # Determine if recall is 1
        if token_metrics['recall'] == 1.0:
            correct_count += 1
        
        # Update totals
        total_precision += token_metrics['precision']
        total_recall += token_metrics['recall']
        total_f1 += token_metrics['f1']
        
        # Print metric summary
        print(f"Metrics:")
        print(f"  - Precision: {token_metrics['precision']:.2f}")
        print(f"  - Recall: {token_metrics['recall']:.2f}")
        print(f"  - F1: {token_metrics['f1']:.2f}")
        
        # Store result
        question_result = {
            "question": user_query,
            "expected_answer": expected_answer,
            "actual_answer": model_answer,
            "precision": token_metrics['precision'],
            "recall": token_metrics['recall'],
            "f1": token_metrics['f1'],
            "sources": sources[:5]  # Store top 5 sources for reference
        }
        results["questions"].append(question_result)
    
    # Calculate overall metrics
    total_questions = len(q_a_pairs)
    results["metrics"] = {
        "total_questions": total_questions,
        "correct_answers": correct_count,
        "accuracy": correct_count / total_questions if total_questions > 0 else 0,
        "precision": total_precision / total_questions if total_questions > 0 else 0,
        "recall": total_recall / total_questions if total_questions > 0 else 0,
        "f1": total_f1 / total_questions if total_questions > 0 else 0,
        "similarity": total_similarity / total_questions if total_questions > 0 else 0
    }
    
    # Print overall results
    print_metrics_summary(results["metrics"], model_name)
    
    # Save results to file
    save_results(results, results_file)
    print(f"\nFull evaluation results saved to {results_file}")

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate LLM performance with test questions')
    parser.add_argument('--data', action='store_true', help='Include latest battery data in the prompt')

    args = parser.parse_args()
    
    # Run evaluation with command line arguments
    main(include_data=args.data)