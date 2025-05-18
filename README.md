# DelphiRAG: Enhancing LLMs Time Serial Inference with Human Experts Study

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

## Abstract

The rapid advancement of large language models (LLMs) has revealed their impressive capabilities in semantic understanding and logical reasoning over text-based data. These developments open new avenues for integrating LLM-based inference with high-value domain knowledge from human experts. However, expert knowledge is often unstructured and limited in volume, particularly within specific industrial domains, rendering traditional fine-tuning approaches impractical.

In this work, we propose a novel and efficient framework that facilitates integrating expert knowledge from a Delphi study into LLMs. We validate our framework through an experiment on a predictive maintenance (PdM) use case involving mobile robotic systems, demonstrating notable inference performance improvements with human experts’ knowledge and the feasibility of leveraging expert knowledge in data-scarce environments.

## 📁 Repository Structure

```text
Knowledge4LLM/
├── doc/              # original knowledge from the website and summarized from the Delphi study
├── include/
│   └── Header file of the publisher
├── launch/
│   ├── old_lidar/
│   │   └── Python files from create3_examples/create3_lidar_slam
│   ├── old_nolidar/
│   │   └── Python files based on create3_examples/create3_lidar_slam with modifications
│   ├── launch_publisher.py (C++ tester)
│   └── Python files (shortcuts) to launch all lidar or all no lidar files
├── run/
│   ├── map.py
│   ├── run_avoider.py
│   └── run_mapper.py
└── src/
    └── Implementation file of the publisher
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Hugging Face Transformers
- FAISS (for retrieval)
- Pandas, NumPy, Scikit-learn
- Jupyter

### Run a example
