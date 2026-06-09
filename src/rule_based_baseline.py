import argparse
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.utils.evaluation_utils import (
    calculate_similarity,
    calculate_token_metrics,
    load_test_questions,
    print_metrics_summary,
    save_results,
)


SEGMENT_PATTERN = re.compile(
    r"Data (?P<signal>.*?), From .*? the minimum value was (?P<min>-?\d+(?:\.\d+)?), "
    r"the maximum value was (?P<max>-?\d+(?:\.\d+)?), and the average value was "
    r"(?P<avg>-?\d+(?:\.\d+)?).*?dominant frequency component was approximately "
    r"(?P<freq>-?\d+(?:\.\d+)?) Hz\. The standard deviation was (?P<std>-?\d+(?:\.\d+)?)\."
    r"(?P<peaks>.*?)(?= Data |$)",
    re.IGNORECASE,
)


def parse_telemetry_description(description: str) -> List[Dict[str, Any]]:
    """Parse engineered telemetry text into numeric signal summaries."""
    segments = []
    for match in SEGMENT_PATTERN.finditer(description):
        peak_text = match.group("peaks") or ""
        peak_count = 0
        peak_match = re.search(r"There were (\d+) peaks detected", peak_text, re.IGNORECASE)
        if peak_match:
            peak_count = int(peak_match.group(1))

        segments.append(
            {
                "signal": match.group("signal").strip(),
                "min": float(match.group("min")),
                "max": float(match.group("max")),
                "avg": float(match.group("avg")),
                "freq": float(match.group("freq")),
                "std": float(match.group("std")),
                "peak_count": peak_count,
            }
        )
    return segments


def load_rule_config(rule_file: str) -> Dict[str, Any]:
    """Load threshold rules from the maintenance knowledge text file."""
    with open(rule_file, "r", encoding="utf-8") as file:
        text = file.read()

    config: Dict[str, Any] = {
        "source_file": rule_file,
        "cliff_thresholds": {},
        "pwm_threshold": None,
        "battery_percentage_threshold": None,
    }

    cliff_names = ["cliff_side_right", "cliff_front_right", "cliff_front_left", "cliff_side_left"]
    for cliff_name in cliff_names:
        match = re.search(rf"{cliff_name}\s+exceeds\s+(\d+(?:\.\d+)?)", text, re.IGNORECASE)
        if match:
            config["cliff_thresholds"][cliff_name] = float(match.group(1))

    pwm_match = re.search(r"pwm_left or pwm_right.*?exceed\s+(\d+(?:\.\d+)?)", text, re.IGNORECASE | re.DOTALL)
    if pwm_match:
        config["pwm_threshold"] = float(pwm_match.group(1))

    battery_match = re.search(r"battery loading percentage is lower than\s+(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if battery_match:
        config["battery_percentage_threshold"] = float(battery_match.group(1))

    if not config["cliff_thresholds"]:
        raise ValueError(f"No cliff thresholds found in {rule_file}")
    if config["pwm_threshold"] is None:
        raise ValueError(f"No PWM threshold found in {rule_file}")
    if config["battery_percentage_threshold"] is None:
        raise ValueError(f"No battery percentage threshold found in {rule_file}")

    return config


def signal_kind(signal_name: str) -> str:
    name = signal_name.lower()
    if "percentage" in name or "%" in name:
        return "battery_percentage"
    if "voltage" in name:
        return "battery_voltage"
    if "cliff" in name:
        return "cliff"
    if "pwm" in name:
        return "pwm"
    return "unknown"


def find_first(segments: List[Dict[str, Any]], kind: str) -> Optional[Dict[str, Any]]:
    for segment in segments:
        if signal_kind(segment["signal"]) == kind:
            return segment
    return None


def cliff_threshold_for_signal(signal_name: str, rules: Dict[str, Any]) -> Optional[float]:
    normalized_signal = signal_name.lower().replace(" ", "_")
    for cliff_name, threshold in rules["cliff_thresholds"].items():
        if cliff_name in normalized_signal:
            return threshold
    if "cliff_front_right" in normalized_signal:
        return rules["cliff_thresholds"].get("cliff_front_right")
    if "cliff_front_left" in normalized_signal:
        return rules["cliff_thresholds"].get("cliff_front_left")
    return None


def classify_segments(segments: List[Dict[str, Any]], rules: Dict[str, Any]) -> Dict[str, Any]:
    """Apply transparent PdM threshold rules to parsed telemetry summaries."""
    findings = []

    battery_percentage = find_first(segments, "battery_percentage")
    battery_threshold = rules["battery_percentage_threshold"]
    if battery_percentage and battery_percentage["min"] < battery_threshold:
        findings.append(
            {
                "type": "battery_low",
                "priority": 3,
                "text": (
                    f"The minimum battery loading percentage is {battery_percentage['min']:.2f}, "
                    f"which is lower than {battery_threshold:.2f}. The reason could be a misconnection with the "
                    "docking station or the robot being stuck somewhere and unable to return to charge."
                ),
            }
        )

    cliff_segments = [s for s in segments if signal_kind(s["signal"]) == "cliff"]
    abnormal_cliffs = []
    for segment in cliff_segments:
        threshold = cliff_threshold_for_signal(segment["signal"], rules)
        if threshold is not None and segment["max"] > threshold:
            abnormal_cliffs.append({**segment, "threshold": threshold})
    if abnormal_cliffs:
        signal = max(abnormal_cliffs, key=lambda item: item["max"])
        findings.append(
            {
                "type": "object_underneath",
                "priority": 2,
                "text": (
                    f"The maximum value of {signal['signal']} is {signal['max']:.2f}, "
                    f"which is higher than {signal['threshold']:.2f}. The reason could be a foreign object stuck "
                    "between the robot and the ground."
                ),
            }
        )

    pwm_segments = [s for s in segments if signal_kind(s["signal"]) == "pwm"]
    pwm_threshold = rules["pwm_threshold"]
    abnormal_pwm = [s for s in pwm_segments if s["max"] > pwm_threshold]
    if abnormal_pwm:
        if len(abnormal_pwm) > 1:
            pwm_text = f"The pwm value of both motors are higher than {pwm_threshold:,.0f}."
        else:
            signal = abnormal_pwm[0]
            pwm_text = f"The maximum value of {signal['signal']} is {signal['max']:.2f}, which is higher than {pwm_threshold:,.0f}."
        findings.append(
            {
                "type": "wheel_entanglement",
                "priority": 2,
                "text": (
                    f"{pwm_text} The reason could be dirt, hair, or small particles sticking to "
                    "the wheel or axle, causing the motor torque to be higher than normal."
                ),
            }
        )

    findings.sort(key=lambda item: item["priority"], reverse=True)
    return {"requires_maintenance": bool(findings), "findings": findings}


def normal_explanation(segments: List[Dict[str, Any]], question: str, rules: Dict[str, Any]) -> str:
    question_lower = question.lower()
    battery_percentage = find_first(segments, "battery_percentage")
    battery_voltage = find_first(segments, "battery_voltage")

    if "voltage" in question_lower and battery_voltage:
        return "The battery voltage will likely continue to decline in the future."
    if battery_percentage:
        battery_threshold = rules["battery_percentage_threshold"]
        return (
            f"The battery loading percentage is {battery_percentage['min']:.2f}, "
            f"which is higher than {battery_threshold:.2f}. The data shows a stable battery percentage with no "
            "significant fluctuations or peaks detected."
        )

    cliff_segments = [s for s in segments if signal_kind(s["signal"]) == "cliff"]
    if cliff_segments:
        side = "right" if any("right" in s["signal"].lower() for s in cliff_segments) else "left"
        return f"The {side} cliff sensor values are within the normal range. The data shows periodic vibration around the average value."

    pwm_segment = find_first(segments, "pwm")
    if pwm_segment:
        if "trend" in question_lower or "future" in question_lower:
            return f"The {pwm_segment['signal']} value is within the normal range. The data changes periodically."
        return (
            f"The {pwm_segment['signal']} value is within the normal range. The reason could be "
            "dirt, hair, or small particles sticking to the wheel or axle, causing the motor "
            "torque to be higher than normal."
        )

    return "No threshold-based maintenance condition is detected from the telemetry summary."


def generate_rule_based_answer(description: str, question: str, rules: Dict[str, Any]) -> Dict[str, Any]:
    segments = parse_telemetry_description(description)
    classification = classify_segments(segments, rules)

    if classification["requires_maintenance"]:
        answer = "The robot requires maintenance. " + " ".join(
            finding["text"] for finding in classification["findings"]
        )
    else:
        subject = "battery" if "battery" in question.lower() else "robot"
        answer = f"The {subject} does not require maintenance. {normal_explanation(segments, question, rules)}"

    return {
        "answer": answer,
        "segments": segments,
        "requires_maintenance": classification["requires_maintenance"],
        "triggered_rules": [finding["type"] for finding in classification["findings"]],
    }


def maintenance_label(text: str) -> Optional[bool]:
    normalized = text.lower()
    if "does not require maintenance" in normalized or "do not require maintenance" in normalized:
        return False
    if "requires maintenance" in normalized or "require maintenance" in normalized:
        return True
    return None


def evaluate_rule_based_baseline(test_file: str, docs_dir: str, output_dir: str, rule_file: str) -> Dict[str, Any]:
    q_a_pairs = load_test_questions(test_file, docs_dir=docs_dir)
    if not q_a_pairs:
        raise ValueError(f"No QA pairs found for {test_file} in {docs_dir}")
    rules = load_rule_config(rule_file)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "rule_based_threshold_baseline"
    results_file = os.path.join(output_dir, f"{test_file}_{model_name}_{timestamp}.json")

    results: Dict[str, Any] = {
        "model": model_name,
        "test_file": test_file,
        "timestamp": timestamp,
        "rule_source_file": rule_file,
        "rule_thresholds": rules,
        "questions": [],
        "metrics": {},
    }

    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_similarity = 0.0
    correct_label_count = 0
    comparable_label_count = 0

    for index, pair in enumerate(q_a_pairs, start=1):
        description = pair.get("description", "")
        question = pair["question"]
        expected_answer = pair["answer"]
        prediction = generate_rule_based_answer(description, question, rules)
        actual_answer = prediction["answer"]

        token_metrics = calculate_token_metrics(expected_answer, actual_answer)
        similarity = calculate_similarity(expected_answer, actual_answer)

        expected_label = maintenance_label(expected_answer)
        predicted_label = maintenance_label(actual_answer)
        label_correct = expected_label is not None and expected_label == predicted_label
        if expected_label is not None and predicted_label is not None:
            comparable_label_count += 1
            if label_correct:
                correct_label_count += 1

        total_precision += token_metrics["precision"]
        total_recall += token_metrics["recall"]
        total_f1 += token_metrics["f1"]
        total_similarity += similarity

        results["questions"].append(
            {
                "index": index,
                "description": description,
                "question": question,
                "expected_answer": expected_answer,
                "actual_answer": actual_answer,
                "precision": token_metrics["precision"],
                "recall": token_metrics["recall"],
                "f1": token_metrics["f1"],
                "similarity": similarity,
                "expected_requires_maintenance": expected_label,
                "predicted_requires_maintenance": predicted_label,
                "label_correct": label_correct,
                "triggered_rules": prediction["triggered_rules"],
                "parsed_segments": prediction["segments"],
            }
        )

    total_questions = len(q_a_pairs)
    results["metrics"] = {
        "total_questions": total_questions,
        "correct_answers": correct_label_count,
        "accuracy": correct_label_count / comparable_label_count if comparable_label_count else 0.0,
        "maintenance_label_accuracy": correct_label_count / comparable_label_count if comparable_label_count else 0.0,
        "maintenance_label_questions": comparable_label_count,
        "precision": total_precision / total_questions,
        "recall": total_recall / total_questions,
        "f1": total_f1 / total_questions,
        "similarity": total_similarity / total_questions,
    }

    save_results(results, results_file)
    print_metrics_summary(results["metrics"], model_name)
    print(f"\nFull rule-based baseline results saved to {results_file}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a transparent rule-based PdM baseline.")
    parser.add_argument("--test-file", default="data_QA", help="QA JSON filename without extension.")
    parser.add_argument("--docs-dir", default="./docs", help="Directory containing QA JSON files.")
    parser.add_argument("--rule-file", default="./docs/robot_knowledge_maintenance.txt", help="Text file used to derive threshold rules.")
    parser.add_argument("--output-dir", default="./evaluation_results", help="Directory for result JSON files.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    evaluate_rule_based_baseline(args.test_file, args.docs_dir, args.output_dir, args.rule_file)


if __name__ == "__main__":
    main()
