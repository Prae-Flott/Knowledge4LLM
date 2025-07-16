# Enhancing LLM Inference with Human Expert Knowledge: A Case Study on Mobile Robotics Fault Diagnosis and Prediction

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Ollama](https://img.shields.io/badge/ollama-white?style=for-the-badge&logo=ollama&logoColor=black)

## Abstract

The rapid advancement of large language models (LLMs) has revealed their impressive capabilities in semantic understanding and logical reasoning over text-based data. These developments open new avenues for integrating LLM-based inference with high-value domain knowledge from human experts. However, expert knowledge is often unstructured and limited in volume, particularly within specific industrial domains, rendering traditional fine-tuning approaches impractical. In this work, we propose a novel and efficient framework that facilitates integrating expert knowledge from a Delphi study into LLMs. We validate our framework through an experiment on a predictive maintenance (PdM) use case involving mobile robotic systems, demonstrating notable inference performance improvements with human experts’ knowledge and the feasibility of leveraging expert knowledge in data-scarce environments.

![Pipeline](/doc/expert_LLM_pipeline.drawio.png)

## 📁 Repository Structure

```text
Knowledge4LLM/
├── doc/                # original knowledge from website and Expert study; and the questions for evaluation
├── knowledge_base/     # embedded knowledge  
├── utils/              # some sub-functions
│   ├── data_to_text.py
│   ├── embedding_utils.py
│   ├── evaluation_utils_llm.py
│   └── evaluation_utils.py
├── data_loader.py      # load the time serial data
├── knowledge_loader.py # load the knowledges in /doc, and vectorized them
├── evaluation.py       # evaluate the performance of the model with a statistical approaches
├── evaluation_llm.py   # evaluate the performance another LLM
└── main.py             # start a chat with the model
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10
- PyTorch
- Ollama
- Pandas, NumPy, Scikit-learn
- Jupyter 

### Run an example

- Run the knowledge-to-RAG transformation script:

```
python knowledge_loader.py
```

- Chat with the model by running:

```
python main.py
```

- evaluate the model performance with statistical approaches by running:

```
python evaluation.py
```

or if you want to evaluate the performance with another LLM

```
python evaluation_llm.py
```

add parameter `--data`, to switch the inference based on the time serial data.
