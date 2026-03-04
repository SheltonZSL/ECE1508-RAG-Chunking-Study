# RAG Chunking Study: How Document Splitting Affects Deep Retrieval-Augmented QA

## Overview
This project studies a key design choice in Retrieval-Augmented Generation (RAG): **chunking** (how we split documents into passages before retrieval). We build a simple RAG question-answering system using **frozen pre-trained deep learning models** (a neural retriever + a generator) and run controlled experiments to answer:

> **How does the way we split documents into chunks change retrieval accuracy and final QA accuracy?**

We will implement multiple chunking strategies, compare against at least one standard public/reference RAG configuration, and produce **reproducible results and analysis**.

---

## Objective
Build a RAG QA pipeline (frozen models only) and evaluate how chunking affects:
- retrieval quality (can we retrieve the right evidence?)
- end-to-end QA quality (does the system answer correctly?)
- efficiency (latency/context usage)

---

## Motivation
Modern QA systems often use deep learning models in two stages:
1. **Neural retriever**: embeds the question and document chunks into vectors and retrieves the most relevant chunks.
2. **Generator (LLM/seq2seq)**: reads retrieved chunks and generates the answer.

In practice, **chunking is critical**:
- chunks too small → lose context, retrieval may miss necessary information
- chunks too large → retrieval becomes noisy, wastes context window

Since we will **not train or fine-tune** any model, our contribution focuses on **deep-learning-driven system behavior**: we treat retriever and generator as fixed deep models and systematically analyze how chunking changes performance, robustness, and efficiency.

---

## Requirements
### 1) Deep-learning-based RAG pipeline (frozen models only)
Implement an end-to-end pipeline with:
- document ingestion
- chunking
- neural dense retrieval (**bi-encoder embeddings + FAISS**)
- answer generation using a pre-trained model

Provide scripts to reproduce:
- indexing
- retrieval
- QA evaluation

### 2) Chunking methods (core design space)
Implement and compare at least three chunking strategies:
- **Fixed-length token chunks** (optional overlap)
- **Structure-based chunks** (paragraph/heading boundaries)
- **Adaptive chunks** (variable size using simple rules such as punctuation/sentence boundaries and length limits)

### 3) Comparison to existing implementation(s)
Benchmark against at least one **standard public/reference RAG configuration**
(default chunking + dense retrieval), and include a classic baseline (**BM25/TF-IDF**) when feasible.

### 4) Detailed experimental analysis (required since no training)
Run controlled ablations on:
- chunk size
- overlap
- top-k retrieval
- chunking strategy

Optionally evaluate:
- reranking vs no reranking

Include:
- quantitative results
- qualitative error analysis with representative examples

### 5) Metrics and reporting (clear and minimal)
Use a public QA benchmark (e.g., **TriviaQA** or **NaturalQuestions**) and report:
- QA quality: **Exact Match (EM)** and **F1**
- Retrieval quality: **Recall@k** and **MRR**
- Efficiency (optional but recommended): average query latency and context length usage

All hyperparameters and settings will be logged for reproducibility.

---

## Milestones
- **Week 5**: Select dataset/corpus; set up repo and baseline RAG pipeline (default chunking + dense retrieval); confirm at least one reference implementation/configuration for comparison.
- **Week 10**: Implement all chunking methods; produce an initial benchmark table (baseline vs ≥2 chunking variants) and at least one ablation result (e.g., chunk size or overlap).
- **Final**: Complete full experimental matrix; finalize benchmark results (EM/F1 + Recall@k/MRR), sensitivity plots, error analysis, and efficiency profiling; submit code, final report, and presentation/demo.

---
