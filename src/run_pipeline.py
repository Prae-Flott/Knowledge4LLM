"""Unified experiment runner organized by the paper pipeline.

The legacy scripts are kept as direct entry points. This file provides a
single CLI that maps those scripts to the manuscript logic.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def enter_project_root() -> None:
    os.chdir(PROJECT_ROOT)
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def build_vector_kb(args: argparse.Namespace) -> None:
    from src.knowledge_loader import load_docs, save_combined_json

    all_docs = load_docs(args.docs_dir, args.output_dir, args.embedding_model)
    total_paragraphs = sum(len(paragraphs) for paragraphs in all_docs.values())
    combined = save_combined_json(all_docs, args.output_dir)

    print("\nVector knowledge-base build complete.")
    print(f"- Documents processed: {len(all_docs)}")
    print(f"- Paragraph chunks: {total_paragraphs}")
    if combined:
        print(f"- Combined file: {combined}")


def build_graph_kb(args: argparse.Namespace) -> None:
    from src.grapgRAG import main as graph_rag_main

    sys.argv = [
        "grapgRAG.py",
        "--docs_dir",
        args.docs_dir,
        "--output_dir",
        args.output_dir,
        "--chunk_size",
        str(args.chunk_size),
        "--chunk_overlap",
        str(args.chunk_overlap),
    ]
    graph_rag_main()


def chat(_: argparse.Namespace) -> None:
    from src.main import main

    main()


def evaluate_rag(args: argparse.Namespace) -> None:
    from src.evaluation import main

    main(include_data=args.data)


def evaluate_llm(args: argparse.Namespace) -> None:
    from src.evaluation_llm import main

    main(include_data=args.data)


def evaluate_graph(args: argparse.Namespace) -> None:
    from src.evaluation_graph import main

    main(include_data=args.data, use_graph=not args.traditional)


def evaluate_threshold(args: argparse.Namespace) -> None:
    from src.rule_based_baseline import evaluate_rule_based_baseline

    os.makedirs(args.output_dir, exist_ok=True)
    evaluate_rule_based_baseline(args.test_file, args.docs_dir, args.output_dir, args.rule_file)


def evaluate_symbolic(args: argparse.Namespace) -> None:
    from src.symbolic_reasoner import evaluate_symbolic_reasoner

    os.makedirs(args.output_dir, exist_ok=True)
    evaluate_symbolic_reasoner(
        args.test_file,
        args.docs_dir,
        args.output_dir,
        args.rule_file,
        args.knowledge_mode,
    )


def evaluate_all_symbolic(args: argparse.Namespace) -> None:
    from src.symbolic_reasoner import evaluate_symbolic_reasoner

    os.makedirs(args.output_dir, exist_ok=True)
    for knowledge_mode in ("expert", "public"):
        for test_file in ("test_QA", "infere_QA", "data_QA"):
            print(f"\n=== KG-R evaluation: {knowledge_mode} / {test_file} ===")
            evaluate_symbolic_reasoner(
                test_file,
                args.docs_dir,
                args.output_dir,
                args.rule_file,
                knowledge_mode,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Knowledge4LLM experiments in the same order as the manuscript pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_vector = subparsers.add_parser("build-vector-kb", help="Build vector chunks from docs/*.txt.")
    build_vector.add_argument("--docs-dir", default="./docs")
    build_vector.add_argument("--output-dir", default="./knowledge_base")
    build_vector.add_argument("--embedding-model", default="nomic-embed-text")
    build_vector.set_defaults(func=build_vector_kb)

    build_graph = subparsers.add_parser("build-graph-kb", help="Build a GraphRAG knowledge base.")
    build_graph.add_argument("--docs-dir", default="./docs")
    build_graph.add_argument("--output-dir", default="./embedding_graph")
    build_graph.add_argument("--chunk-size", type=int, default=500)
    build_graph.add_argument("--chunk-overlap", type=int, default=50)
    build_graph.set_defaults(func=build_graph_kb)

    chat_parser = subparsers.add_parser("chat", help="Run an interactive expert-grounded LLM query.")
    chat_parser.set_defaults(func=chat)

    rag = subparsers.add_parser("evaluate-rag", help="Evaluate expert-grounded RAG inference.")
    rag.add_argument("--data", action="store_true", help="Include telemetry descriptions.")
    rag.set_defaults(func=evaluate_rag)

    llm = subparsers.add_parser("evaluate-llm", help="Evaluate LLM-only or alternative LLM inference.")
    llm.add_argument("--data", action="store_true", help="Include telemetry descriptions.")
    llm.set_defaults(func=evaluate_llm)

    graph = subparsers.add_parser("evaluate-graph", help="Evaluate GraphRAG retrieval.")
    graph.add_argument("--data", action="store_true", help="Include telemetry descriptions.")
    graph.add_argument("--traditional", action="store_true", help="Use traditional vector retrieval.")
    graph.set_defaults(func=evaluate_graph)

    threshold = subparsers.add_parser("evaluate-threshold", help="Evaluate telemetry threshold rules.")
    threshold.add_argument("--test-file", default="data_QA")
    threshold.add_argument("--docs-dir", default="./docs")
    threshold.add_argument("--rule-file", default="./docs/robot_knowledge_maintenance.txt")
    threshold.add_argument("--output-dir", default="./evaluation_results")
    threshold.set_defaults(func=evaluate_threshold)

    symbolic = subparsers.add_parser("evaluate-symbolic", help="Evaluate the KG-R baseline.")
    symbolic.add_argument("--test-file", default="data_QA")
    symbolic.add_argument("--docs-dir", default="./docs")
    symbolic.add_argument("--rule-file", default="./docs/robot_knowledge_maintenance.txt")
    symbolic.add_argument("--output-dir", default="./evaluation_results")
    symbolic.add_argument("--knowledge-mode", choices=["expert", "public"], default="expert")
    symbolic.set_defaults(func=evaluate_symbolic)

    all_symbolic = subparsers.add_parser("evaluate-all-symbolic", help="Run all KG-R table evaluations.")
    all_symbolic.add_argument("--docs-dir", default="./docs")
    all_symbolic.add_argument("--rule-file", default="./docs/robot_knowledge_maintenance.txt")
    all_symbolic.add_argument("--output-dir", default="./evaluation_results")
    all_symbolic.set_defaults(func=evaluate_all_symbolic)

    return parser


def main() -> None:
    enter_project_root()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
