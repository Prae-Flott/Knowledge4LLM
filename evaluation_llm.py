import os
import re
import json
import ollama
import time
from datetime import datetime
from typing import Dict, List, Any

# Import project modules
from data_loader import read_latest_description
from utils.embedding_utils import get_knowledge_context
from utils.evaluation_utils import (
    load_test_questions,
    save_results
)
from utils.evaluation_utils_llm import (
    SemanticEvaluator
)

def main(include_data: bool=False) -> str:
    """
    Run inference with model and perform per-question semantic evaluation.
        
    Returns:
    --------
    str:
        Path to the evaluation results file
    """
    model_name = "qwen2.5:32b"
    test_file = "test_QA"
    knowledge_dir = "./knowledge_base"
    top_k = 10
    output_dir = "./evaluation_results"
    evaluator_model = "gemma3:4b"

    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{output_dir}/infere_llm_no_know_eval_{model_name.replace(':', '_')}_{timestamp}.json"
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    SYSTEM_PROMPT = (
        "You are a helpful mobile robot fault prediction and diagnosis assistant who answers questions based on snippets of text provided in context."
        "Answer only using the context provided, being as concise as possible. If you're unsure, just say that you don't know.\n"
        "Instructions: Give a brief, to-the-point answer. Keep your answer as short as possible.\n"
        "Context:\n\n"
    )

    # Add latest battery data if requested
    if include_data:
        results_file = f"{output_dir}/llm_eval_with_data_{model_name.replace(':', '_')}_{timestamp}.json"
        print(f"Including latest battery data in evaluation")
            

    # Load test questions
    q_a_pairs = load_test_questions(test_file, docs_dir="./docs")
    if not q_a_pairs:
        print("No test questions found. Exiting.")
        return ""
    
    print(f"Starting evaluation with {len(q_a_pairs)} questions using model {model_name}...")
    
    # Initialize results storage
    results = {
        "model": model_name,
        "evaluator_model": evaluator_model,
        "test_file": test_file,
        "timestamp": timestamp,
        "questions": [],
        "metrics": {}
    }
    
    # Track metrics
    correct_count = 0
    semantic_correctness_total = 0
    semantic_completeness_total = 0
    semantic_conciseness_total = 0
    semantic_overall_total = 0
    total_time = 0
    total_tokens = 0
    
    # Initialize semantic evaluator
    evaluator = SemanticEvaluator(evaluator_model=evaluator_model, output_dir=output_dir)
    
    # Process each question-answer pair
    for i, pair in enumerate(q_a_pairs):
        user_query = pair["question"]
        
        # Add data to the query if requested
        if include_data:
            robo_data = pair["description"]
            user_query = f"Given this robot running data: '{robo_data}', please answer: {user_query}"
        
        expected_answer = pair["answer"]
        
        print(f"\n[{i+1}/{len(q_a_pairs)}] Q: {user_query}")
        print(f"Expected: {expected_answer}")

        # Retrieve relevant context from knowledge base
        context, sources = get_knowledge_context(
            user_query, 
            knowledge_dir=knowledge_dir, 
            modelname="nomic-embed-text", 
            top_k=top_k)

        # Start timing
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
        
        # Remove the thinking part from the answer
        if "<think>\n" in model_answer and "</think>\n" in model_answer:    
            actual_answer = re.sub(r'<think>\n.*?</think>\n\n', '', model_answer, flags=re.DOTALL)
        else:
            actual_answer = model_answer
        print(f"Model: {actual_answer}")
        
        # Count tokens in the response
        token_count = len(actual_answer.split())
        tokens_per_second = token_count / generation_time if generation_time > 0 else 0
        time_per_token = generation_time / token_count if token_count > 0 else 0
        
        # Perform semantic evaluation for this question
        print("Evaluating answer semantically...")
        semantic_eval = evaluator.evaluate_answer(expected_answer, actual_answer)
        
        # Update semantic metrics totals
        semantic_correctness_total += semantic_eval['semantic_correctness']
        semantic_completeness_total += semantic_eval['completeness']
        semantic_conciseness_total += semantic_eval['conciseness']
        semantic_overall_total += semantic_eval['overall_score']
        
        # Consider answer correct if overall semantic score is high enough
        if semantic_eval['overall_score'] >= 0.8:
            correct_count += 1
        
        # Update time and token totals
        total_time += generation_time
        total_tokens += token_count
        
        # Print metrics summary
        print(f"Generation time: {generation_time:.2f}s for {token_count} tokens")
        print(f"Time per token: {time_per_token*1000:.2f} ms")
        print(f"Tokens per second: {tokens_per_second:.2f}")
        print(f"Semantic scores:")
        print(f"  - Correctness: {semantic_eval['semantic_correctness']:.2f}")
        print(f"  - Completeness: {semantic_eval['completeness']:.2f}")
        print(f"  - Conciseness: {semantic_eval['conciseness']:.2f}")
        print(f"  - Overall: {semantic_eval['overall_score']:.2f}")
        
        # Store result with semantic evaluation
        question_result = {
            "question": user_query,
            "expected_answer": expected_answer,
            "actual_answer": actual_answer,
            "generation_time": generation_time,
            "token_count": token_count,
            "tokens_per_second": tokens_per_second,
            "time_per_token": time_per_token,
            "sources": sources[:5],  # Store top 5 sources for reference
            "semantic_evaluation": semantic_eval
        }
        results["questions"].append(question_result)
    
    # Calculate overall metrics
    total_questions = len(q_a_pairs)
    
    # Calculate average time per token
    avg_time_per_token = total_time / total_tokens if total_tokens > 0 else 0
    avg_tokens_per_second = total_tokens / total_time if total_time > 0 else 0
    
    # Calculate average semantic scores
    avg_semantic_correctness = semantic_correctness_total / total_questions if total_questions > 0 else 0
    avg_semantic_completeness = semantic_completeness_total / total_questions if total_questions > 0 else 0
    avg_semantic_conciseness = semantic_conciseness_total / total_questions if total_questions > 0 else 0
    avg_semantic_overall = semantic_overall_total / total_questions if total_questions > 0 else 0
    
    # Add all metrics to results
    results["metrics"] = {
        "total_questions": total_questions,
        "correct_answers": correct_count,
        "accuracy": correct_count / total_questions if total_questions > 0 else 0,
        "semantic_correctness": avg_semantic_correctness,
        "semantic_completeness": avg_semantic_completeness,
        "semantic_conciseness": avg_semantic_conciseness,
        "semantic_overall": avg_semantic_overall,
        "total_generation_time": total_time,
        "total_tokens_generated": total_tokens,
        "avg_time_per_token": avg_time_per_token,
        "avg_time_per_token_ms": avg_time_per_token * 1000,
        "avg_tokens_per_second": avg_tokens_per_second
    }
    
    # Print overall results
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total questions: {total_questions}")
    print(f"Correct answers: {correct_count}")
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"\nSemantic Metrics:")
    print(f"  - Correctness: {avg_semantic_correctness:.4f}")
    print(f"  - Completeness: {avg_semantic_completeness:.4f}")
    print(f"  - Conciseness: {avg_semantic_conciseness:.4f}")
    print(f"  - Overall Score: {avg_semantic_overall:.4f}")
    print(f"\nGeneration Performance:")
    print(f"  - Total time: {total_time:.2f}s")
    print(f"  - Total tokens: {total_tokens}")
    print(f"  - Avg. time per token: {avg_time_per_token*1000:.2f} ms")
    print(f"  - Avg. tokens per second: {avg_tokens_per_second:.2f}")
    
    # Save results to file
    save_results(results, results_file)
    print(f"\nFull evaluation results saved to {results_file}")
    
    return results_file

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate LLM performance with test questions')
    parser.add_argument('--data', action='store_true', help='Include latest battery data in the prompt')

    args = parser.parse_args()
        
    # Run evaluation with command line arguments
    main(include_data=args.data)