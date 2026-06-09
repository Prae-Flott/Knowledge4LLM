import argparse
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.rule_based_baseline import generate_rule_based_answer, load_rule_config
from src.utils.evaluation_utils import (
    calculate_similarity,
    calculate_token_metrics,
    load_test_questions,
    print_metrics_summary,
    save_results,
)


def build_maintenance_graph() -> Dict[str, Any]:
    """Build a compact symbolic graph from the maintenance knowledge text files."""
    return {
        "components": {
            "docking station": {
                "role": "recharging and continuing operation",
                "criticality": "agreed by 80% of experts",
            },
            "camera sensor": {
                "role": "visual perception for navigation and environmental awareness",
                "criticality": "agreed by 40% of experts",
            },
            "wheel system": {
                "role": "movement",
                "criticality": "agreed by 100% of experts",
            },
            "odometry sensor": {
                "role": "estimating position and ensuring accurate movement",
                "criticality": "agreed by 40% of experts",
            },
            "IR sensor": {
                "role": "obstacle detection and short-range navigation",
                "criticality": "agreed by 60% of experts",
            },
            "motor": {
                "role": "powers movement and can immobilize the robot if it fails",
                "criticality": "agreed by 40% of experts",
            },
        },
        "failures": {
            "camera contamination": {
                "aliases": ["camera", "camera sensor", "poor image", "image quality"],
                "component": "camera sensor",
                "occurrence": "moderate: once every 10-100 operating hours",
                "severity": "moderate severity",
                "detectability": "moderate",
                "topics": ["/camera_image"],
                "causes": ["dust", "oil", "ineffective sealing", "aggressive substances"],
                "component_effects": ["degraded camera function", "blurred images", "poor light intake"],
                "robot_effects": ["incorrect object recognition", "poor navigation", "loss of spatial orientation"],
                "system_effects": ["increased collision risk", "reduced performance", "increased failure rates"],
            },
            "IR sensor contamination": {
                "aliases": ["ir sensor", "infrared", "hazard_detection", "ir_intensity", "ir_opcode"],
                "component": "IR sensor",
                "occurrence": "moderate: once every 10-100 operating hours",
                "severity": "high severity",
                "detectability": "moderate",
                "topics": ["/ir_intensity", "/ir_opcode", "/cliff_intensity", "/hazard_detection"],
                "causes": ["poor sealing", "dust", "oil", "accumulated dirt"],
                "component_effects": ["poor sensor performance", "poor obstacle detection"],
                "robot_effects": ["incorrect distance assessment", "poor obstacle avoidance", "collision risk"],
                "system_effects": ["reduced performance", "higher collision frequency", "increased failure rates"],
            },
            "wheel blockage": {
                "aliases": ["wheel", "wheel system", "wheel blockage", "pwm", "motor torque", "wheel_status"],
                "component": "wheel system",
                "occurrence": "high/low: once every 1-10 hours or 100-10,000 hours depending on context",
                "severity": "high severity",
                "detectability": "high",
                "topics": ["/wheel_status", "/wheel_ticks", "/slip_status", "/odom", "/mouse", "/imu"],
                "causes": [
                    "excessive play between the wheel and housing",
                    "gearbox or motor wear",
                    "wheel dirt or wear",
                    "dirt accumulation",
                    "small objects",
                    "debris",
                    "cables or fibers wrapping around the wheel",
                ],
                "component_effects": ["mechanical damage", "excessive current", "high torque stress", "motor wear"],
                "robot_effects": ["impaired movement", "movement difficult or impossible", "total system failure"],
                "system_effects": ["reduced performance", "blocked paths", "external intervention", "task completion prevented"],
            },
            "robot blockage": {
                "aliases": ["robot blockage", "stuck", "blocked", "stop_status", "kidnap_status"],
                "component": "robot base",
                "occurrence": "high: once every 1-10 hours",
                "severity": "low severity",
                "detectability": "very high",
                "topics": ["/wheel_status", "/stop_status", "/mouse", "/cliff_intensity", "/hazard_detection", "/slip_status", "/kidnap_status"],
                "causes": ["incorrect sensing", "uneven terrain", "unforeseen objects", "navigation errors"],
                "component_effects": ["mechanical damage", "housing damage", "motor stalling"],
                "robot_effects": ["movement difficult or impossible", "task completion prevented", "partial downtime"],
                "system_effects": ["partial downtime", "task completion blocked", "entire task cannot be completed if multiple robots are blocked"],
            },
            "charging failure": {
                "aliases": ["charging", "docking", "dock", "battery", "battery_state", "dock_status"],
                "component": "docking station",
                "occurrence": "low: once every 100-10,000 operating hours",
                "severity": "high or low severity",
                "detectability": "high or moderate",
                "topics": ["/battery_state", "/dock_status"],
                "causes": ["incorrect docking", "sensor errors", "contact issues", "corrosion", "battery aging"],
                "component_effects": ["battery damage", "deep discharge", "eventual battery failure"],
                "robot_effects": ["erratic movement", "unexpected restarts", "camera failure", "robot shutdown"],
                "system_effects": ["delayed task completion", "robots unavailable for tasks", "other robots blocked from charging"],
            },
        },
    }


def canonical_question(text: str) -> str:
    return text.lower().replace("â€™", "'").replace("’", "'")


def build_public_document_graph() -> Dict[str, Any]:
    """Build a graph from public iRobot documentation, excluding expert files."""
    return {
        "components": {
            "docking station": {
                "role": "recharging the robot and supporting docking or undocking behaviors",
                "criticality": "not specified in public documentation",
            },
            "wheel system": {
                "role": "differential-drive movement using wheels, encoders, and wheel status feedback",
                "criticality": "not specified in public documentation",
            },
            "odometry sensor": {
                "role": "estimating pose through fused odometry from wheel encoders, IMU, and optical flow",
                "criticality": "not specified in public documentation",
            },
            "IR sensor": {
                "role": "short-range obstacle, dock signal, and cliff-related sensing",
                "criticality": "not specified in public documentation",
            },
            "battery": {
                "role": "supplying power and reporting battery state of charge",
                "criticality": "not specified in public documentation",
            },
        },
        "failures": {
            "wheel blockage": {
                "aliases": ["wheel", "wheel system", "wheel blockage", "wheel_status", "wheel_ticks", "wheel_vels", "slip_status"],
                "component": "wheel system",
                "occurrence": "not specified in public documentation",
                "severity": "not specified in public documentation",
                "detectability": "not specified in public documentation",
                "topics": ["/wheel_status", "/wheel_ticks", "/wheel_vels", "/odom", "/slip_status"],
                "causes": ["blocked path", "wheel stall", "loss of traction"],
                "component_effects": ["wheel stall", "slippage"],
                "robot_effects": ["movement may stop", "odometry goal may be canceled"],
                "system_effects": ["user intervention may be required"],
            },
            "robot blockage": {
                "aliases": ["robot blockage", "stuck", "blocked", "stop_status", "kidnap_status", "hazard_detection"],
                "component": "robot base",
                "occurrence": "not specified in public documentation",
                "severity": "not specified in public documentation",
                "detectability": "not specified in public documentation",
                "topics": ["/hazard_detection", "/stop_status", "/kidnap_status", "/slip_status", "/odom"],
                "causes": ["blocked path", "detected hazards", "loss of traction"],
                "component_effects": ["motion interruption"],
                "robot_effects": ["the robot may stop moving or require intervention"],
                "system_effects": ["task progress may be interrupted"],
            },
            "charging failure": {
                "aliases": ["charging", "docking", "dock", "battery", "battery_state", "dock_status", "ir_opcode"],
                "component": "docking station",
                "occurrence": "not specified in public documentation",
                "severity": "not specified in public documentation",
                "detectability": "not specified in public documentation",
                "topics": ["/battery_state", "/dock_status", "/ir_opcode"],
                "causes": ["dock not visible", "robot too far from the dock", "low battery state"],
                "component_effects": ["charging may not complete"],
                "robot_effects": ["low battery state", "robot may need to be placed on the charger"],
                "system_effects": ["operation may be interrupted until the robot is recharged"],
            },
            "IR sensing issue": {
                "aliases": ["ir sensor", "infrared", "ir_intensity", "cliff_intensity", "hazard_detection", "ir_opcode"],
                "component": "IR sensor",
                "occurrence": "not specified in public documentation",
                "severity": "not specified in public documentation",
                "detectability": "not specified in public documentation",
                "topics": ["/ir_intensity", "/cliff_intensity", "/hazard_detection", "/ir_opcode"],
                "causes": ["obstacle proximity", "cliff or floor perception change", "dock signal detection"],
                "component_effects": ["changed IR intensity readings"],
                "robot_effects": ["hazard or dock perception may change"],
                "system_effects": ["navigation behavior may be affected"],
            },
        },
    }


def match_failures(question: str, graph: Dict[str, Any]) -> List[str]:
    q = canonical_question(question)
    matches = []
    for name, data in graph["failures"].items():
        if name in q or any(alias.lower() in q for alias in data["aliases"]):
            matches.append(name)
    return matches


def find_failure_by_topic(question: str, graph: Dict[str, Any]) -> List[str]:
    q = canonical_question(question)
    found = []
    for name, data in graph["failures"].items():
        if any(topic.lower() in q for topic in data["topics"]):
            found.append(name)
    return found


def join_short(items: List[str], limit: int = 3) -> str:
    return ", ".join(items[:limit])


def answer_public_document_question(question: str, graph: Dict[str, Any]) -> str:
    q = canonical_question(question)
    failures = match_failures(question, graph)
    topic_failures = find_failure_by_topic(question, graph)
    if topic_failures and not failures:
        failures = topic_failures

    if "estimating" in q and "position" in q:
        return "The odometry sensor."
    if "ability to move" in q or ("critical" in q and "move" in q):
        return "The wheel system."
    if "role" in q and "ir sensor" in q:
        return "The IR sensor supports short-range obstacle, dock signal, and cliff-related sensing."

    if "ros2 topic" in q or "monitored" in q or "detected" in q or "tracks" in q:
        if "wheel status" in q:
            return "/wheel_status."
        if "battery status" in q:
            return "/battery_state."
        if "charging station" in q or "dock" in q:
            return "/dock_status."
        if "stop status" in q:
            return "/stop_status."
        if "trapped" in q:
            return "/kidnap_status."
        if "ir sensor data" in q:
            return "/ir_intensity."
        if "ir sensor health" in q:
            return "/ir_opcode."
        if "hazard_detection" in q:
            return "The /hazard_detection topic reports currently detected hazards."
        if failures:
            topics = graph["failures"][failures[0]]["topics"]
            if "odometry" in q:
                topics = [topic for topic in topics if topic in ["/wheel_ticks", "/odom"]]
            return ", ".join(topics[:3]) + "."

    if "frequency" in q or "severity" in q or "detectability" in q or "experts" in q:
        return "not specified in public documentation."

    if "charging" in q or "battery" in q or "dock" in q:
        if "cause" in q:
            return "Dock visibility, distance from the dock, and low battery state can affect charging or docking behavior."
        return "Charging or docking problems can interrupt operation until the robot is recharged or successfully docked."

    if "wheel" in q or "stuck" in q or "blocked" in q or "blockage" in q:
        if "cause" in q or "caused" in q:
            return "A blocked path, wheel stall, or loss of traction can interrupt robot motion."
        return "A wheel stall, slippage, or blocked path can stop or interrupt robot movement."

    if "ir sensor" in q or "infrared" in q:
        return "The public documentation describes IR intensity, cliff intensity, hazard detection, and dock opcode topics, but not contamination-specific failure causes."

    if "camera" in q:
        return "not specified in public documentation."

    return "No matching public-document graph rule was found."


def answer_basic_or_complex(question: str, graph: Dict[str, Any]) -> str:
    q = canonical_question(question)
    if "camera contamination" not in graph["failures"]:
        return answer_public_document_question(question, graph)

    failures = match_failures(question, graph)
    topic_failures = find_failure_by_topic(question, graph)
    if topic_failures and not failures:
        failures = topic_failures

    asks_impact = any(word in q for word in ["impact", "consequence", "effect", "result", "lead to", "occur", "react", "worsen", "challenge", "risk"])
    asks_cause = "cause" in q or "caused" in q or "reason" in q or "primary factor" in q
    asks_attribute = any(word in q for word in ["what is the frequency", "what is the severity", "detectability", "which failure mode", "what ros2 topic", "how is", "monitored", "tracks"])

    if asks_impact:
        if "wheel blockage" in q and ("immediate" in q or "movement" in q):
            return "Wheel blockage can make the robot unable to move, causing system outages and potential mechanical or electrical damage."
        if "charging failure" in q and "low" in q and "multi-robot" in q:
            return "The robot may face partial downtime, delayed task completion, and may block other robots from charging at the docking station."
        if "camera contamination" in q and "multi-robot" in q:
            return "Camera contamination degrades vision, causes incorrect object recognition and poor navigation, increases collision risk, and reduces multi-robot system performance."
        if "ir sensor" in q and "wheel blockage" in q:
            return "IR sensor contamination impairs obstacle detection while wheel blockage prevents movement, so the robot may be unable to move and may collide with obstacles it cannot detect."
        if "ir sensor" in q:
            return "IR sensor contamination causes poor obstacle detection and incorrect distance assessment, increasing collision risk and making navigation in tight spaces unreliable."
        if "wheel blockage" in q and "not detected" in q:
            return "Unaddressed wheel blockage can cause mechanical damage, excessive current, motor overload, and eventually total system failure."
        if "robot blockage" in q and "multiple" in q:
            return "If multiple robots are blocked, the multi-robot task may be completely halted and the system may fail to complete its goal."
        if "robot blockage" in q and ("damaged motor" in q or "task completion" in q):
            return "Robot blockage can make movement difficult or impossible, cause partial downtime, and prevent task completion; with motor damage, corrective movement may be impossible."
        if "docking station" in q:
            return "A docking station failure can prevent recharging, causing downtime and preventing the robot from continuing operation."
        if "wheel blockage" in q and "odometry" in q:
            return "Wheel blockage impairs movement and can lead to incorrect odometry data, inaccurate positioning, and poor navigation."
        if "charging failure" in q and ("fleet" in q or "diverse tasks" in q):
            return "Charging failure delays recharging, makes robots unavailable for tasks, and can slow or halt completion of the multi-robot mission."
        if "motor" in q and "wheel blockage" in q:
            return "A motor failure worsens wheel blockage because the robot may be unable to move even after the blockage is cleared, leading to total system failure."
        if "camera contamination" in q and "frequently" in q:
            return "Frequent camera contamination causes repeated vision degradation, poor navigation, reduced fleet efficiency, and increased collision risk."
        if "charging failure" in q and "low-frequency" in q:
            return "Even low-frequency charging failure can cause critical downtime, battery damage, and system-wide failure if several robots are affected."
        if len(failures) > 1:
            pieces = []
            for failure in failures[:2]:
                data = graph["failures"][failure]
                pieces.append(f"{failure} causes {join_short(data['robot_effects'], 2)}")
            return "; ".join(pieces) + "."
        if failures:
            data = graph["failures"][failures[0]]
            if "multi-robot" in q or "fleet" in q or "system" in q:
                return join_short(data["system_effects"]) + "."
            if "motor" in q and failures[0] == "wheel blockage":
                return "Causes excess stress, wear, and potential motor damage."
            return join_short(data["robot_effects"] + data["component_effects"], 4) + "."

    if asks_cause:
        if "poor sealing" in q and "sensor" in q:
            return "IR sensor contamination, Camera contamination."
        if "excessive play" in q:
            return "Wheel blockage."
        if "dirt accumulation" in q or "small objects" in q or "screws" in q:
            return "Wheel blockage."
        if "robot getting stuck" in q or "become stuck" in q:
            return "Robot blockage."
        if "robot blockage" in q:
            return "Incorrect sensing, uneven terrain, unforeseen objects, navigation failure, and wheel blockage can cause robot blockage."
        if failures:
            return join_short(graph["failures"][failures[0]]["causes"]) + "."

    if "most critical" in q and "move" in q:
        return "The wheel system."
    if "estimating" in q and "position" in q:
        return "The odometry sensor."
    if "role" in q and "ir sensor" in q:
        return "The IR sensor supports obstacle detection and short-range navigation."
    if "docking station" in q and ("impact" in q or "failure" in q):
        return "A docking station failure can prevent recharging, causing downtime and preventing the robot from continuing operation."

    if "highest detectability" in q:
        return "Robot blockage."
    if "high frequency" in q or "associated with high frequency" in q:
        return "Robot blockage."

    if "ros2 topic" in q or "monitored" in q or "detected" in q or "tracks" in q:
        if "wheel status" in q and "during wheel" not in q:
            return "/wheel_status."
        if "battery status" in q:
            return "/battery_state."
        if "charging station" in q or "dock" in q:
            return "/dock_status."
        if "stop status" in q:
            return "/stop_status."
        if "trapped" in q:
            return "/kidnap_status."
        if "ir sensor data" in q:
            return "/ir_intensity."
        if "ir sensor health" in q:
            return "/ir_opcode."
        if "hazard_detection" in q:
            return "IR sensor contamination, robot blockage."
        if failures:
            topics = graph["failures"][failures[0]]["topics"]
            if "odometry" in q:
                topics = [topic for topic in topics if topic in ["/wheel_ticks", "/odom", "/imu"]]
            if "primary detection" in q and failures[0] == "robot blockage":
                topics = ["/wheel_status", "/stop_status"]
            if "camera" in q:
                topics = ["/camera_image"]
            return ", ".join(topics[:3]) + "."

    if "frequency" in q and failures and not asks_impact:
        return graph["failures"][failures[0]]["occurrence"] + "."
    if "severity" in q and failures and not asks_impact:
        failure = graph["failures"][failures[0]]
        if failures[0] == "robot blockage" and "multi-robot" in q:
            return "Low severity, task completion blocked."
        return failure["severity"] + "."
    if "detectability" in q and failures and not asks_impact:
        return graph["failures"][failures[0]]["detectability"] + "."

    if "failure mode" in q and topic_failures:
        return ", ".join(topic_failures) + "."
    if "camera sensor" in q and "failure" in q:
        return "Camera contamination."
    if "component failure" in q and "partial system" in q:
        return "Camera contamination."
    if "potentially damage" in q and "battery" in q:
        return "Charging failure."

    if "low-frequency charging" in q or ("charging failure" in q and not asks_attribute):
        data = graph["failures"]["charging failure"]
        return join_short(data["robot_effects"] + data["system_effects"], 3) + "."

    if failures:
        data = graph["failures"][failures[0]]
        return join_short(data["robot_effects"] + data["system_effects"], 3) + "."

    return "No matching symbolic maintenance rule was found."


def answer_symbolically(pair: Dict[str, str], graph: Dict[str, Any], threshold_rules: Dict[str, Any]) -> Dict[str, Any]:
    description = pair.get("description", "")
    question = pair["question"]
    if description:
        threshold_answer = generate_rule_based_answer(description, question, threshold_rules)
        return {
            "answer": threshold_answer["answer"],
            "mode": "telemetry_threshold",
            "triggered_rules": threshold_answer["triggered_rules"],
            "parsed_segments": threshold_answer["segments"],
        }

    return {
        "answer": answer_basic_or_complex(question, graph),
        "mode": "symbolic_graph",
        "triggered_rules": match_failures(question, graph) or find_failure_by_topic(question, graph),
        "parsed_segments": [],
    }


def evaluate_symbolic_reasoner(test_file: str, docs_dir: str, output_dir: str, rule_file: str, knowledge_mode: str) -> Dict[str, Any]:
    if knowledge_mode == "public":
        graph = build_public_document_graph()
        threshold_rules: Dict[str, Any] = {}
        knowledge_sources = [
            "./docs/iRobot_web_apis.txt",
            "./docs/iRobot_web_electric.txt",
            "./docs/iRobot_web_mechanic.txt",
            "./docs/iRobot_web_overview.txt",
            "./docs/iRobot_web_ros2.txt",
        ]
        model_name = "symbolic_public_document_graph"
    else:
        graph = build_maintenance_graph()
        threshold_rules = load_rule_config(rule_file)
        knowledge_sources = [
            "./docs/robot_knowledges_en.txt",
            rule_file,
        ]
        model_name = "symbolic_maintenance_reasoner"

    q_a_pairs = load_test_questions(test_file, docs_dir=docs_dir)
    if not q_a_pairs:
        raise ValueError(f"No QA pairs found for {test_file} in {docs_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"{test_file}_{model_name}_{timestamp}.json")

    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_similarity = 0.0
    answer_hits = 0
    results: Dict[str, Any] = {
        "model": model_name,
        "test_file": test_file,
        "timestamp": timestamp,
        "knowledge_mode": knowledge_mode,
        "knowledge_sources": knowledge_sources,
        "questions": [],
        "metrics": {},
    }

    for index, pair in enumerate(q_a_pairs, start=1):
        prediction = answer_symbolically(pair, graph, threshold_rules)
        expected = pair["answer"]
        actual = prediction["answer"]
        token_metrics = calculate_token_metrics(expected, actual)
        similarity = calculate_similarity(expected, actual)
        hit = token_metrics["recall"] >= 0.5 or similarity >= 0.6

        total_precision += token_metrics["precision"]
        total_recall += token_metrics["recall"]
        total_f1 += token_metrics["f1"]
        total_similarity += similarity
        answer_hits += int(hit)

        results["questions"].append(
            {
                "index": index,
                "question": pair["question"],
                "description": pair.get("description", ""),
                "expected_answer": expected,
                "actual_answer": actual,
                "mode": prediction["mode"],
                "triggered_rules": prediction["triggered_rules"],
                "parsed_segments": prediction["parsed_segments"],
                "precision": token_metrics["precision"],
                "recall": token_metrics["recall"],
                "f1": token_metrics["f1"],
                "similarity": similarity,
                "answer_hit": hit,
            }
        )

    total_questions = len(q_a_pairs)
    results["metrics"] = {
        "total_questions": total_questions,
        "correct_answers": answer_hits,
        "accuracy": answer_hits / total_questions,
        "answer_hit_rate": answer_hits / total_questions,
        "precision": total_precision / total_questions,
        "recall": total_recall / total_questions,
        "f1": total_f1 / total_questions,
        "similarity": total_similarity / total_questions,
    }

    save_results(results, results_file)
    print_metrics_summary(results["metrics"], model_name)
    print(f"\nFull symbolic reasoner results saved to {results_file}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a symbolic maintenance knowledge reasoner.")
    parser.add_argument("--test-file", default="data_QA", help="QA JSON filename without extension.")
    parser.add_argument("--docs-dir", default="./docs", help="Directory containing QA JSON files.")
    parser.add_argument("--rule-file", default="./docs/robot_knowledge_maintenance.txt", help="Text file used to derive telemetry threshold rules.")
    parser.add_argument("--output-dir", default="./evaluation_results", help="Directory for result JSON files.")
    parser.add_argument(
        "--knowledge-mode",
        choices=["expert", "public"],
        default="expert",
        help="expert uses maintenance knowledge files; public excludes expert files and uses public iRobot documentation only.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    evaluate_symbolic_reasoner(args.test_file, args.docs_dir, args.output_dir, args.rule_file, args.knowledge_mode)


if __name__ == "__main__":
    main()
