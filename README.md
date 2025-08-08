# 📘 ATF: Adaptive Table Filtering Framework for Table-based Question Answering

**[📄 Paper (arXiv)](https://arxiv.org/abs/2506.23463)**  
A modular and extensible pre-processing framework for improving TableQA performance and input efficiency.

---

## ✨ Overview

**ATF (Adaptive Table Filtering)** is a plug-and-play framework that enhances table-based question answering by **removing irrelevant columns and rows** before passing the compact table into LLMs.

It fuses **LLM-based reasoning** and **retrieval-based techniques**, making it adaptable, efficient, and accurate — especially under long-table or limited-token constraints.

---

## 🧠 Architecture

The system is modularized into the following components:

1. **Answer Entity Prediction (LLM)**
   - Predicts the entity type of the expected answer to guide the filtering process.

2. **Column Relevance Scoring**
   - Uses LLM-generated descriptions, cosine similarity, and ensemble scoring to select relevant columns.

3. **Column Clustering**
   - Clusters columns using K-means and retains the core cluster plus top-ranked others.

4. **Row Ranking**
   - Computes row relevance using TF-IDF, BM25, and Sentence-BERT embeddings.

5. **Final Table Selector**
   - Combines selected columns and rows to build a compact context for the LLM to use.

---

## ⚙️ Features

- 🧠 **Intent-aware filtering** based on question types (Who/When/etc.)
- 🔍 **Hybrid retrieval** (dense + sparse)
- 🧮 **K-means clustering** for semantic column grouping
- 📦 **Cosine similarity cache** for repeated queries
- 💬 **LangGraph agent-style orchestration** (coming soon)
- 🧪 **Failure logging & few-shot adaptation roadmap**

---


## 📊 System Flowchart

```mermaid
graph TD
    A[data_loader]
    B[answer_entity_predictor]
    C[column_relevance_checker]
    D[column_cluster_agent]
    E[essential_column_node]
    F[select_column_agent]
    G[row_ranker]
    H[final_table_selector]
    I[responder]
    J[END]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
```

---

## 📊 Detailed Modules

<details>
<summary><strong>Entity Type Prediction</strong></summary>

```mermaid
graph TD
    P1[Input: Question] --> P2[LLM CoT Reasoning]
    P2 --> P3[Score Each Entity Type]
    P3 --> P4[Select Most Probable Type]
    P4 --> P5[Output: predicted_answer_entity]
```

</details>

<details>
<summary><strong>Column Relevance & Clustering</strong></summary>

```mermaid
graph TD
    A1[Input: Question + Raw Table]
    A1 --> A2[LLM Column Descriptions]
    A2 --> A3[LLM Scoring]
    A1 --> A4[Cosine Similarity]
    A3 --> A5[Fusion Score]
    A4 --> A5
    A5 --> A6[Clustering (K-means)]
    A6 --> A7[Output: Selected Columns]
```

</details>

<details>
<summary><strong>Cluster Selection (Ensemble)</strong></summary>

```mermaid
graph TD
    B1[Input: clusters, scores, question]
    B1 --> B2[Semantic Similarity]
    B1 --> B3[MCDM Scoring]
    B1 --> B4[Adaptive Threshold]
    B2 --> B5[Voting]
    B3 --> B5
    B4 --> B5
    B5 --> B6[Select top-k]
    B6 --> B7[Output: Filtered Columns]
```

</details>

<details>
<summary><strong>Row Ranking</strong></summary>

```mermaid
graph TD
    R1[Input: selected_columns + question]
    R1 --> R2[Join rows to text]
    R2 --> R3[TF-IDF]
    R2 --> R4[BM25]
    R2 --> R5[Sentence-BERT]
    R3 --> R6[Softmax]
    R4 --> R6
    R5 --> R6
    R6 --> R7[Weighted Fusion]
    R7 --> R8[Top-k Rows]
    R8 --> R9[Output: Final Table Rows]
```

</details>

---

## 🧪 Citation

If you use this framework or paper, please cite:

```bibtex
@misc{jang2025dropadaptivetablefiltering,
      title={What to Keep and What to Drop: Adaptive Table Filtering Framework}, 
      author={WonJune Jang},
      year={2025},
      eprint={2506.23463},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.23463}, 
}
```

---

## 🙋‍♀️ Contact

For questions, contributions, or collaborations, feel free to reach out at:  
📧 **dnjswnswkd03@mju.ac.kr**

---
