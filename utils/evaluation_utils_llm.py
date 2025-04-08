import os
import re
import json
import ollama
import time
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Import project modules
from utils.text_spliter import parse_file
from data_loader import read_latest_description
from utils.embedding_utils import get_knowledge_context
from utils.evaluation_utils import (
    load_test_questions,
    normalize_text,
    calculate_token_metrics,
    save_results,
    print_metrics_summary
)

class SemanticEvaluator:
    """
    Uses an LLM to perform semantic evaluation of model outputs against expected answers.
    """
    
    def __init__(self, evaluator_model: str = "deepseek-r1:1.5b", output_dir: str = "./evaluation_results"):
        """
        Initialize the semantic evaluator.
        
        Parameters:
        -----------
        evaluator_model : str
            The model to use for evaluation (should be different from the model being evaluated)
        output_dir : str
            Directory to save evaluation results
        """
        self.evaluator_model = evaluator_model
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Define the evaluation prompt template
        self.EVALUATOR_PROMPT = """
        You are an expert evaluator of question answering systems. Your task is to compare a model's answer to an expected answer and evaluate if they are semantically equivalent.

        Expected Answer: "{expected_answer}"
        Model Answer: "{model_answer}"

        Please evaluate the two answers based on the following criteria:
        1. Semantic Correctness: Are the core facts and information in the model's answer correct compared to the expected answer?
        2. Completeness: Does the model's answer include all the important information from the expected answer?
        3. Conciseness: Is the model's answer free from irrelevant or incorrect information?

        Provide your evaluation in the following JSON format:
        {{
        "semantic_correctness": (score between 0.0 and 1.0),
        "completeness": (score between 0.0 and 1.0),
        "conciseness": (score between 0.0 and 1.0),
        "overall_score": (average of the above scores),
        "reasoning": "Brief explanation of your scoring"
        }}

        Only respond with the JSON, no additional text.
        """

    def evaluate_answer(self, expected_answer: str, model_answer: str) -> Dict[str, Any]:
        """
        Use the LLM to evaluate semantic equivalence between expected and model answers.
        
        Parameters:
        -----------
        expected_answer : str
            The expected (ground truth) answer
        model_answer : str
            The model's predicted answer
            
        Returns:
        --------
        Dict[str, Any]:
            Evaluation results including semantic scores
        """
        # Skip evaluation if answers are empty
        if not expected_answer or not model_answer:
            return {
                "semantic_correctness": 0.0,
                "completeness": 0.0,
                "conciseness": 0.0,
                "overall_score": 0.0,
                "reasoning": "One or both answers are empty"
            }
        
        # Format the evaluator prompt
        prompt = self.EVALUATOR_PROMPT.format(
            expected_answer=expected_answer,
            model_answer=model_answer
        )
        
        try:
            # Get evaluation from the LLM
            response = ollama.chat(
                model=self.evaluator_model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of question answering systems."},
                    {"role": "user", "content": prompt}
                ],
            )
            
            # Extract and parse the JSON response
            evaluation_text = response["message"]["content"].strip()
            
            # Clean up the response to extract just the JSON part
            if "```json" in evaluation_text:
                evaluation_text = evaluation_text.split("```json")[1].split("```")[0].strip()
            elif "```" in evaluation_text:
                evaluation_text = evaluation_text.split("```")[1].split("```")[0].strip()
            
            evaluation = json.loads(evaluation_text)
            
            # Ensure all required fields are present
            required_fields = ["semantic_correctness", "completeness", "conciseness", "overall_score", "reasoning"]
            for field in required_fields:
                if field not in evaluation:
                    evaluation[field] = 0.0 if field != "reasoning" else "Field missing in LLM response"
            
            return evaluation
            
        except Exception as e:
            print(f"Error during semantic evaluation: {e}")
            # Return default values in case of error
            return {
                "semantic_correctness": 0.0,
                "completeness": 0.0,
                "conciseness": 0.0,
                "overall_score": 0.0,
                "reasoning": f"Error in evaluation: {str(e)}"
            }

    def evaluate_results(self, results_file: str) -> str:
        """
        Perform semantic evaluation on existing results.
        
        Parameters:
        -----------
        results_file : str
            Path to the JSON file with previous evaluation results
            
        Returns:
        --------
        str:
            Path to the newly created semantic evaluation results file
        """
        try:
            # Load existing results
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            model_name = results.get('model', 'unknown')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{self.output_dir}/semantic_eval_{model_name.replace(':', '_')}_{timestamp}.json"
            
            print(f"Performing semantic evaluation of {len(results.get('questions', []))} questions...")
            print(f"Using evaluator model: {self.evaluator_model}")
            
            # Add semantic evaluation to each question
            for i, question in enumerate(results.get('questions', [])):
                expected_answer = question.get('expected_answer', '')
                model_answer = question.get('actual_answer', '')
                
                print(f"\nEvaluating question {i+1}/{len(results.get('questions', []))}")
                print(f"Q: {question.get('question', '')}")
                
                # Perform semantic evaluation
                semantic_eval = self.evaluate_answer(expected_answer, model_answer)
                
                # Add semantic evaluation to the question results
                question['semantic_evaluation'] = semantic_eval
                
                # Print evaluation summary
                print(f"Semantic evaluation:")
                print(f"  - Correctness: {semantic_eval['semantic_correctness']:.2f}")
                print(f"  - Completeness: {semantic_eval['completeness']:.2f}")
                print(f"  - Conciseness: {semantic_eval['conciseness']:.2f}")
                print(f"  - Overall: {semantic_eval['overall_score']:.2f}")
                print(f"  - Reasoning: {semantic_eval['reasoning']}")
            
            # Calculate aggregate semantic metrics
            semantic_correctness = sum(q['semantic_evaluation']['semantic_correctness'] 
                                      for q in results['questions']) / len(results['questions'])
            completeness = sum(q['semantic_evaluation']['completeness'] 
                              for q in results['questions']) / len(results['questions'])
            conciseness = sum(q['semantic_evaluation']['conciseness'] 
                             for q in results['questions']) / len(results['questions'])
            overall_score = sum(q['semantic_evaluation']['overall_score'] 
                               for q in results['questions']) / len(results['questions'])
            
            # Add semantic metrics to overall results
            results['metrics']['semantic_correctness'] = semantic_correctness
            results['metrics']['semantic_completeness'] = completeness
            results['metrics']['semantic_conciseness'] = conciseness
            results['metrics']['semantic_overall'] = overall_score
            
            # Add evaluator model info
            results['semantic_evaluator'] = {
                'model': self.evaluator_model,
                'timestamp': timestamp
            }
            
            # Save enhanced results
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            
            print("\n" + "="*50)
            print("SEMANTIC EVALUATION SUMMARY")
            print("="*50)
            print(f"Semantic Correctness: {semantic_correctness:.4f}")
            print(f"Semantic Completeness: {completeness:.4f}")
            print(f"Semantic Conciseness: {conciseness:.4f}")
            print(f"Semantic Overall Score: {overall_score:.4f}")
            print(f"\nResults saved to {output_file}")
            
            return output_file
            
        except Exception as e:
            print(f"Error during semantic evaluation: {e}")
            return ""
