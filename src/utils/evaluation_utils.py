import os
import json
import re
from difflib import SequenceMatcher
from typing import Dict, List, Any, Tuple, Set, Optional

def load_test_questions(file_name: str, docs_dir: str = './docs') -> List[Dict[str, str]]:
    """
    Load question-answer pairs from a JSON file.
    
    Parameters:
    -----------
    file_name : str
        Filename without extension (e.g., 'test_QA')
    docs_dir : str
        Directory where the JSON files are stored
        
    Returns:
    --------
    list or None:
        List of question-answer pairs if successful, None if file not found
    """
    q_a_pairs = []

    try:
        # Construct the expected file path
        file_path = os.path.join(docs_dir, f"{file_name}.json")
        
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
        
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        
        # Check if content is a list or single item
        if isinstance(content, list):
            # Process each item in the list
            for item in content:
                if "question" in item and "answer" in item:
                    q_a_pairs.append(item)
        elif isinstance(content, dict) and "question" in content and "answer" in content:
            # Process a single item
            q_a_pairs.append(content)
    
        print(f"Loaded {len(q_a_pairs)} test questions from {file_path}")
        return q_a_pairs
    except Exception as e:
        print(f"Error reading test questions: {e}")
        return q_a_pairs


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison by removing punctuation,
    converting to lowercase, and standardizing whitespace.
    
    Parameters:
    -----------
    text : str
        Text to normalize
        
    Returns:
    --------
    str:
        Normalized text
    """
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """
    Tokenize text into words.
    
    Parameters:
    -----------
    text : str
        Text to tokenize
        
    Returns:
    --------
    List[str]:
        List of tokens (words)
    """
    return normalize_text(text).split()


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate string similarity ratio between two texts.
    
    Parameters:
    -----------
    text1 : str
        First text
    text2 : str
        Second text
        
    Returns:
    --------
    float:
        Similarity score between 0.0 and 1.0
    """
    return SequenceMatcher(None, normalize_text(text1), normalize_text(text2)).ratio()


def calculate_token_metrics(reference: str, prediction: str) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score based on token overlap.
    
    Parameters:
    -----------
    reference : str
        The reference (expected) answer
    prediction : str
        The predicted answer from the LLM
        
    Returns:
    --------
    Dict:
        Dictionary with precision, recall, and f1 metrics
    """
    # Tokenize both strings
    ref_tokens = set(tokenize(reference))
    pred_tokens = set(tokenize(prediction))
    
    if not ref_tokens:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    if not pred_tokens:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # Calculate true positives (tokens in both sets)
    true_positives = len(ref_tokens.intersection(pred_tokens))
    
    # Calculate precision: TP / (TP + FP)
    precision = true_positives / len(pred_tokens) if pred_tokens else 0
    
    # Calculate recall: TP / (TP + FN)
    recall = true_positives / len(ref_tokens) if ref_tokens else 0
    
    # Calculate F1 score: 2 * (precision * recall) / (precision + recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def save_results(results: Dict[str, Any], filename: str) -> None:
    """
    Save evaluation results to a JSON file.
    
    Parameters:
    -----------
    results : dict
        The evaluation results to save
    filename : str
        The filename to save to
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
    
    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filename}")


def print_metrics_summary(metrics: Dict[str, Any], model_name: str) -> None:
    """
    Print a summary of evaluation metrics.
    
    Parameters:
    -----------
    metrics : Dict[str, Any]
        Dictionary containing evaluation metrics
    model_name : str
        Name of the model being evaluated
    """
    print("\n" + "="*50)
    print(f"EVALUATION SUMMARY FOR {model_name}")
    print("="*50)
    
    # Print count metrics
    total_questions = metrics.get('total_questions', 0)
    correct_answers = metrics.get('correct_answers', 0)
    print(f"Total Questions: {total_questions}")
    print(f"Correct Answers: {correct_answers}")
    
    # Print score metrics
    print(f"Accuracy: {metrics.get('accuracy', 0):.4f} ({correct_answers}/{total_questions})")
    print(f"Average Precision: {metrics.get('precision', 0):.4f}")
    print(f"Average Recall: {metrics.get('recall', 0):.4f}")
    print(f"Average F1 Score: {metrics.get('f1', 0):.4f}")
    print(f"Average Similarity: {metrics.get('similarity', 0):.4f}")


def visualize_results(results: Dict[str, Any], output_dir: str = './evaluation_results') -> str:
    """
    Create visualizations of evaluation results.
    
    Parameters:
    -----------
    results : Dict[str, Any]
        Evaluation results to visualize
    output_dir : str
        Directory to save visualizations
        
    Returns:
    --------
    str:
        Path to the saved visualization file
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from datetime import datetime
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract data for visualization
        metrics = results.get('metrics', {})
        questions = results.get('questions', [])
        model_name = results.get('model', 'unknown')
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Overall metrics (bar chart)
        metric_names = ['accuracy', 'precision', 'recall', 'f1', 'similarity']
        metric_values = [metrics.get(name, 0) for name in metric_names]
        
        axes[0, 0].bar(metric_names, metric_values, color='skyblue')
        axes[0, 0].set_ylim(0, 1.05)
        axes[0, 0].set_title('Overall Metrics')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot 2: Precision vs Recall (scatter plot)
        precision_scores = [q.get('precision', 0) for q in questions]
        recall_scores = [q.get('recall', 0) for q in questions]
        
        axes[0, 1].scatter(recall_scores, precision_scores, alpha=0.7, color='green')
        axes[0, 1].set_title('Precision vs Recall')
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_xlim(0, 1.05)
        axes[0, 1].set_ylim(0, 1.05)
        axes[0, 1].grid(alpha=0.3)
        
        # Plot 3: F1 Scores (histogram)
        f1_scores = [q.get('f1', 0) for q in questions]
        
        axes[1, 0].hist(f1_scores, bins=10, color='purple', alpha=0.7)
        axes[1, 0].set_title('Distribution of F1 Scores')
        axes[1, 0].set_xlabel('F1 Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(alpha=0.3)
        
        # Plot 4: Correct vs Incorrect (pie chart)
        correct_count = sum(1 for q in questions if q.get('is_correct', False))
        incorrect_count = len(questions) - correct_count
        
        axes[1, 1].pie([correct_count, incorrect_count], 
                      labels=['Correct', 'Incorrect'], 
                      colors=['green', 'red'],
                      autopct='%1.1f%%',
                      startangle=90)
        axes[1, 1].set_title('Accuracy')
        
        # Set overall title
        plt.suptitle(f'Evaluation Results for {model_name}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save the figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"viz_{model_name.replace(':', '_')}_{timestamp}.png")
        plt.savefig(filename, dpi=300)
        plt.close()
        
        print(f"Visualization saved to {filename}")
        return filename
        
    except ImportError:
        print("Matplotlib not installed. Skipping visualization.")
        return ""