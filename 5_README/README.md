# Conversational AI — Assignment 2: Hybrid RAG System with Automated Evaluation

## Group 122

| S.No | Name | Email | Contribution |
|------|------|-------|--------------|
| 1 | SK SHAHRUKH SABA | 2024aa05401@wilp.bits-pilani.ac.in | 100% |
| 2 | SANKHA CHAKRABORTY | 2024AA05393@wilp.bits-pilani.ac.in | 100% |
| 3 | NEELASHA ROY | 2024aa05698@wilp.bits-pilani.ac.in | 100% |
| 4 | ARUNAVA DUTTA | 2024aa05374@wilp.bits-pilani.ac.in | 100% |
| 5 | CHINTAN DESAI | 2024aa05648@wilp.bits-pilani.ac.in | 100% |

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Prerequisites and Installation](#2-prerequisites-and-installation)
   - [System Requirements](#2a-system-requirements)
   - [Environment Setup](#2b-environment-setup)
   - [Dependencies Table](#2c-dependencies-table)
   - [Troubleshooting](#2d-troubleshooting)
3. [How to Run (Quick Start)](#3-how-to-run-quick-start)
   - [One-Command Execution](#3a-one-command-execution-powershell-scripts)
   - [Manual Step-by-Step](#3b-manual-step-by-step)
   - [Expected Execution Times](#3c-expected-execution-times)
4. [Project File Map](#4-project-file-map)
5. [Part 1 — Detailed Walkthrough](#5-part-1--detailed-walkthrough)
   - [Section 0: Wikipedia Dataset Generation](#5a-section-0-wikipedia-dataset-generation)
   - [Section 1.1: Dense Vector Retrieval](#5b-section-11-dense-vector-retrieval)
   - [Section 1.2: Sparse Keyword Retrieval (BM25)](#5c-section-12-sparse-keyword-retrieval-bm25)
   - [Section 1.3: Reciprocal Rank Fusion (RRF)](#5d-section-13-reciprocal-rank-fusion-rrf)
   - [Section 1.4: Response Generation](#5e-section-14-response-generation)
   - [Section 1.5: Streamlit Web Application](#5f-section-15-streamlit-web-application)
6. [Part 2 — Detailed Walkthrough](#6-part-2--detailed-walkthrough)
   - [Section 2.1: Question Generation](#6a-section-21-question-generation)
   - [Section 2.2: Metrics Implementation](#6b-section-22-metrics-implementation)
   - [Section 2.4: Evaluation Pipeline](#6c-section-24-evaluation-pipeline)
   - [Evaluation Dashboard](#6d-evaluation-dashboard)
7. [Evaluation Report](#7-evaluation-report)
   - [Overall Performance Summary](#7a-overall-performance-summary)
   - [Metric Justifications](#7b-metric-justifications)
   - [MRR Documentation](#7c-mrr-mandatory-metric-documentation)
   - [Results Table](#7d-results-table)
   - [Visualizations](#7e-visualizations)
   - [Ablation Study Results](#7f-ablation-study-results)
   - [LLM-as-Judge Results](#7g-llm-as-judge-results)
   - [Error Analysis](#7h-error-analysis)
8. [Data Files](#8-data-files)
   - [Fixed URLs (200)](#8a-fixed-urls-200--full-list)
   - [Random URLs (300)](#8b-random-urls-300)
   - [Evaluation Questions (100)](#8c-evaluation-questions-100)
   - [JSON Schema Reference](#8d-json-schema-reference)
   - [Output Files](#8e-output-files)
9. [Technical Notes](#9-technical-notes)
   - [Architecture Decisions](#9a-architecture-decisions)
   - [Limitations](#9b-limitations)
   - [Reproduction Instructions](#9c-reproduction-instructions)
   - [References](#9d-references)
10. [Challenges and Learning Experience](#10-challenges-and-learning-experience)
    - [Environment Setup Challenges](#10a-environment-setup-challenges)
    - [Model-Related Challenges](#10b-model-related-challenges)
    - [Evaluation Pipeline Challenges](#10c-evaluation-pipeline-challenges)
    - [Integration Challenges](#10d-integration-challenges)
    - [Key Learnings](#10e-key-learnings)
    - [What I Would Do Differently](#10f-what-i-would-do-differently)
    - [Time Investment](#10g-time-investment)
    - [Skills Developed](#10h-skills-developed)

---

## 1. Project Overview

This project implements a complete **Hybrid Retrieval-Augmented Generation (RAG)** system with an **automated evaluation framework**. The system retrieves relevant context from a corpus of 500 Wikipedia articles using a combination of dense semantic search and sparse keyword matching, fused via Reciprocal Rank Fusion (RRF), and then generates answers using a locally-running TinyLlama-1.1B-Chat language model. The evaluation framework automatically generates 100 Q&A pairs across four question categories, evaluates the RAG pipeline using three complementary metrics (MRR, ROUGE-L, BERTScore), runs an ablation study comparing five retrieval configurations, and produces a comprehensive HTML report with visualizations.

**Part 1**: Hybrid RAG pipeline combining Dense Retrieval (FAISS + all-MiniLM-L6-v2) and Sparse Retrieval (BM25) via Reciprocal Rank Fusion (RRF), with response generation using TinyLlama-1.1B-Chat (4-bit GGUF quantization).

**Part 2**: Automated evaluation framework generating 100 Q&A pairs (factual, comparative, inferential, multi-hop), computing three metrics (MRR, ROUGE-L, BERTScore), running a 5-configuration ablation study, and producing a full HTML evaluation report.
**Important Note**: 1. We have used a json logic to ensure the urls are truely unique.
                  2. We have included important learnings and challenges faced during the project at the end of the report.
### Architecture Diagram

```
User Query
  |
  +---> Dense Retrieval (FAISS + all-MiniLM-L6-v2)
  |       top-K results
  |
  +---> Sparse Retrieval (BM25)
  |       top-K results
  |
  +---> Reciprocal Rank Fusion (RRF, k=60)
          top-N combined results
              |
              +---> TinyLlama-1.1B-Chat (4-bit GGUF)
                      Generated Answer
```

---

## 2. Prerequisites and Installation

### 2a. System Requirements

- **Python**: 3.10+ (tested with 3.14.2)
- **OS**: Windows 10/11 (PowerShell instructions provided)
- **RAM**: 8 GB minimum
- **Disk**: ~2 GB (models + virtual environment)
- **Internet**: Required for model downloads and Wikipedia API access

### 2b. Environment Setup

```powershell
# Step 1: Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Step 2: Install all dependencies
pip install -r requirements.txt

# Step 3: Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
```

> **Note**: If you encounter an execution policy error, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 2c. Dependencies Table

| Package | Min Version | Purpose | Used By |
|---------|-------------|---------|---------|
| requests | >=2.31.0 | HTTP requests, Wikipedia API | Both |
| beautifulsoup4 | >=4.12.0 | HTML parsing fallback | Both |
| certifi | >=2023.7.22 | SSL certificates | Both |
| sentence-transformers | >=2.2.2 | Dense embeddings (all-MiniLM-L6-v2) | Both |
| faiss-cpu | >=1.7.4 | Vector similarity search index | Both |
| numpy | >=1.24.0 | Numerical operations | Both |
| rank-bm25 | >=0.2.2 | BM25 sparse retrieval | Both |
| nltk | >=3.8.1 | Tokenization and stopword removal | Both |
| ctransformers | >=0.2.27 | TinyLlama GGUF inference | Both |
| huggingface_hub | >=0.17.0 | Model downloading | Both |
| streamlit | >=1.28.0 | Web UI (Part 1 app + Part 2 dashboard) | Both |
| rouge-score | >=0.1.2 | ROUGE-L metric | Part 2 |
| bert-score | >=0.3.13 | BERTScore metric | Part 2 |
| pandas | >=2.0.0 | Data tables and CSV output | Part 2 |
| scikit-learn | >=1.3.0 | ML utilities | Part 2 |
| matplotlib | >=3.7.0 | Chart generation for HTML report | Part 2 |
| seaborn | >=0.12.0 | Statistical visualization | Part 2 |
| plotly | >=5.18.0 | Interactive charts in dashboard | Part 2 |
| tqdm | >=4.66.0 | Progress bars | Both |
| statsmodels | >=0.14.0 | Statistical modeling | Part 2 |

### 2d. Troubleshooting

**SSL Certificate Errors**

```python
import certifi, os
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['CURL_CA_BUNDLE'] = certifi.where()
```

**Memory Issues**

- Close other applications
- Reduce `top_k` parameter in retrieval functions
- Use smaller batch sizes for embedding generation

**Model Download Failures**

```powershell
pip install huggingface_hub
huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

**NLTK Data Missing**

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
```

**Virtual Environment Activation (ExecutionPolicy)**

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 3. How to Run (Quick Start)

### 3a. One-Command Execution (PowerShell Scripts)

```powershell
# Part 1: Generate datasets + launch RAG app
.\run_part1.ps1

# Part 2: Full evaluation pipeline + dashboard
.\run_part2.ps1

# Part 2 with skip options
.\run_part2.ps1 -SkipUrls -SkipQuestions    # Use existing data
.\run_part2.ps1 -OnlyDashboard              # Just view results
.\run_part2.ps1 -NoDashboard                # Skip dashboard launch
```

### 3b. Manual Step-by-Step

```powershell
# 1. Generate Wikipedia URL datasets (200 fixed + 300 random)
python generate_urls.py --fixed 200 --random 300 --group GROUP_122

# 2. Run the notebook (VS Code / Jupyter)
#    Open ConversationalAI_Assignment_2_Group_122_Par1_1.1_to_1.4.ipynb
#    and execute all cells sequentially

# 3. Launch Part 1 Streamlit app
streamlit run ConversationalAI_Assignment_2_Group_122_Par1_1.5-streamlit-app.py

# 4. Generate 100 evaluation Q&A pairs
python ConversationalAI_Assignment_2_Group_122_Part2_QuestionGen.py

# 5. Run the full evaluation pipeline
python ConversationalAI_Assignment_2_Group_122_Part2_Evaluation.py

# 6. Launch the evaluation dashboard
streamlit run ConversationalAI_Assignment_2_Group_122_Part2_Dashboard.py
```

### 3c. Expected Execution Times

| Step | Script | Estimated Time |
|------|--------|----------------|
| URL generation | `generate_urls.py` | 10-20 min |
| Notebook (full) | `...Par1_1.1_to_1.4.ipynb` | 25-35 min |
| Question generation | `Part2_QuestionGen.py` | 15-30 min |
| Evaluation pipeline | `Part2_Evaluation.py` | 30-60 min |
| Dashboard | `Part2_Dashboard.py` | Instant (persistent) |

---

## 4. Project File Map

```
Assignment 2/
|-- README.md                           <-- This file
|-- requirements.txt                    <-- All Python dependencies
|-- CLAUDE.md                           <-- Claude Code instructions
|
|-- PART 1: Hybrid RAG System
|   |-- ConversationalAI_..._Par1_1.1_to_1.4.ipynb   <-- Core RAG notebook
|   |-- ConversationalAI_..._Par1_1.5-streamlit-app.py <-- Interactive RAG UI
|   |-- generate_urls.py                              <-- URL dataset generator (CLI)
|   +-- run_part1.ps1                                 <-- Part 1 orchestration script
|
|-- PART 2: Automated Evaluation
|   |-- ConversationalAI_..._Part2_QuestionGen.py     <-- Q&A pair generator
|   |-- ConversationalAI_..._Part2_Metrics.py         <-- Metric implementations
|   |-- ConversationalAI_..._Part2_Evaluation.py      <-- Evaluation pipeline
|   |-- ConversationalAI_..._Part2_Dashboard.py       <-- Evaluation dashboard
|   +-- run_part2.ps1                                 <-- Part 2 orchestration script
|
|-- DATA FILES
|   |-- fixed_url.json              <-- 200 curated Wikipedia articles
|   |-- random_url.json             <-- 300 random Wikipedia articles
|   +-- evaluation_questions.json   <-- 100 Q&A pairs for evaluation
|
+-- OUTPUT FILES (generated by Part 2 pipeline)
    |-- evaluation_results.csv      <-- Per-question metrics
    |-- ablation_results.json       <-- 5-config ablation comparison
    +-- evaluation_report.html      <-- Full HTML report with charts
```

---

## 5. Part 1 -- Detailed Walkthrough

### 5a. Section 0: Wikipedia Dataset Generation

**Script**: `generate_urls.py` (standalone CLI) + notebook cells

**Purpose**: Build a corpus of 500 Wikipedia articles for the RAG system.

- **Fixed set (200)**: Hardcoded curated URLs across 8 domains (people, places, events, technology, science, arts, sports, organization)
- **Random set (300)**: MediaWiki API `list=random`, filtered for minimum 200 words per article
- **Hash-based deterministic seeding**: `hash("GROUP_122")` ensures the fixed set is reproducible
- **Domain classification**: Keyword matching on Wikipedia categories
- **JSON schema**: `[{id, url, domain, content}]`
- **Fallback content extraction**: API plaintext extract, then BeautifulSoup HTML scraping
- **Rate limiting**: 1-second delay between API batches, exponential backoff on retries

**Functions in `generate_urls.py`:**

| Function | Purpose |
|----------|---------|
| `api_request(session, params, max_retries)` | MediaWiki API call with retry and exponential backoff |
| `get_random_article_titles(count, session)` | Batch random titles via `list=random` (50 per batch) |
| `get_article_data_batch(titles, session)` | Batch extracts + categories (20 per batch) |
| `classify_domain(categories, title)` | Keyword-based domain classification |
| `extract_content_api(extract_text)` | Clean API plaintext extract (remove reference markers) |
| `extract_content_bs4(url, session)` | Fallback HTML scraping via BeautifulSoup |
| `generate_fixed_dataset(count, group_id, session)` | Build fixed set from hardcoded URL list |
| `generate_random_dataset(count, group_id, fixed_urls, session)` | Build random set via API |

### 5b. Section 1.1: Dense Vector Retrieval

- **Embedding model**: `all-MiniLM-L6-v2` (384 dimensions, 22M parameters)
- **Index**: FAISS `IndexFlatIP` (inner product on L2-normalized vectors = cosine similarity)
- **Pipeline**: Encode query -> normalize -> FAISS search -> return top-K with similarity scores

**Cosine similarity formula:**

```
cosine_similarity(q, d) = (q . d) / (||q|| * ||d||)
```

After L2 normalization: `cosine_similarity = inner_product(q_norm, d_norm)`

**Key functions**: `load_sentence_transformer()`, `build_faiss_index()`, `retrieve_top_k_dense()`

### 5c. Section 1.2: Sparse Keyword Retrieval (BM25)

- **Algorithm**: BM25Okapi (k1=1.5, b=0.75)
- **Preprocessing**: lowercase -> remove punctuation -> NLTK tokenize -> remove stopwords

**BM25 formula:**

```
BM25(D, Q) = SUM IDF(q_i) * [f(q_i,D) * (k1+1)] / [f(q_i,D) + k1 * (1 - b + b * |D|/avgdl)]
```

**Key functions**: `build_bm25_index()`, `retrieve_top_k_bm25()`

### 5d. Section 1.3: Reciprocal Rank Fusion (RRF)

- **Purpose**: Combine dense + sparse rankings into a single hybrid result set
- **Formula**: `RRF_score(d) = SUM 1/(k + rank_i(d))` where k=60 (default)
- **Algorithm**: Retrieve top-K from each method -> compute RRF score per document -> sort -> return top-N

**Function**: `reciprocal_rank_fusion(query, embedding_model, faiss_index, bm25_index, corpus, preprocess_func, top_k=10, top_n=5, rrf_k=60)`

**Output fields per result**: rank, rrf_score, document, dense_rank, dense_score, bm25_rank, bm25_score

### 5e. Section 1.4: Response Generation

- **LLM**: TinyLlama-1.1B-Chat (4-bit GGUF, Q4_K_M variant, ~700 MB)
- **Loader**: ctransformers `AutoModelForCausalLM`
- **Prompt format**: ChatML (`<|system|>`, `<|user|>`, `<|assistant|>`)
- **Generation parameters**: max_tokens=80, temperature=0.1, top_p=0.9, top_k=40, repetition_penalty=1.1
- **Context**: top-N RRF chunks concatenated, max 600 characters

**Prompt template:**

```
<|system|>
You are a helpful assistant. Answer questions using ONLY the provided context.
If the context doesn't contain the answer, say "I cannot answer based on the provided context."
Be concise and direct.</s>
<|user|>
Context: {context}

Question: {query}</s>
<|assistant|>
Answer:
```

**Key functions**: `build_context_from_chunks()`, `create_prompt()`, `generate_response()`

**Return dict**: query, retrieved_chunks, dense_results, bm25_results, context, answer, timing

### 5f. Section 1.5: Streamlit Web Application

- **Script**: `ConversationalAI_Assignment_2_Group_122_Par1_1.5-streamlit-app.py`
- **Purpose**: Interactive web UI for querying the RAG system
- **Layout**: Sidebar (config sliders: top_k, top_n, max_tokens, temperature) + Main area (query input, answer, chunks, timing)
- **Caching**: `@st.cache_resource` for models/indices, `@st.cache_data` for corpus
- **Features**: Expandable chunk sections with all 3 scores (dense, BM25, RRF), source URL links, response time breakdown (retrieval vs generation)
- **Run**: `streamlit run ConversationalAI_Assignment_2_Group_122_Par1_1.5-streamlit-app.py` -> opens at http://localhost:8501

---

## 6. Part 2 -- Detailed Walkthrough

### 6a. Section 2.1: Question Generation

**Script**: `ConversationalAI_Assignment_2_Group_122_Part2_QuestionGen.py`

**Goal**: Generate 100 Q&A pairs from the 500-document corpus.

**Distribution**:
- 40 Factual questions (direct facts from documents)
- 20 Comparative questions (comparing entities/concepts)
- 25 Inferential questions (requiring reasoning)
- 15 Multi-hop questions (requiring multiple document pieces)

**Method**: TinyLlama with category-specific prompt templates + fallback templates when LLM generation fails.

**Prompt templates (summary):**

| Category | Prompt Goal | Fallback Example |
|----------|-------------|------------------|
| Factual | Ask about a specific fact, date, name, or definition | "What is {entity} known for?" |
| Comparative | Compare aspects, features, or characteristics from text | "How does {entity} compare to others in its field?" |
| Inferential | Require understanding and reasoning about the text | "Why is {entity} significant?" |
| Multi-hop | Connect multiple pieces of information from text | "What is {entity} and how is it connected to its field?" |

**Functions:**

| Function | Purpose |
|----------|---------|
| `load_llm_model()` | Load TinyLlama via ctransformers |
| `load_corpus()` | Load fixed_url.json + random_url.json |
| `extract_entity_from_url(url)` | Extract article title from Wikipedia URL |
| `generate_question_with_llm(model, content, category)` | LLM-based Q&A generation with ChatML prompts |
| `generate_fallback_question(doc, category)` | Template-based fallback when LLM fails |
| `generate_questions(model, corpus, distribution)` | Full pipeline: iterate categories, validate, collect |

**Output schema** (`evaluation_questions.json`):

```json
{
  "id": "q001",
  "question": "What is X known for?",
  "ground_truth": "X is...",
  "source_url": "https://en.wikipedia.org/wiki/X",
  "source_id": "uuid",
  "category": "factual",
  "domain": "people"
}
```

### 6b. Section 2.2: Metrics Implementation

**Script**: `ConversationalAI_Assignment_2_Group_122_Part2_Metrics.py`

See [Section 7 (Evaluation Report)](#7-evaluation-report) for full metric documentation including formulas, justifications, and interpretation guidelines.

**Functions:**

| Function | Signature | Returns |
|----------|-----------|---------|
| `calculate_mrr` | `(questions, rag_results)` | `{mrr, interpretation, reciprocal_ranks, ranks, statistics, details}` |
| `calculate_rouge_l` | `(generated, references)` | `{mean_f1, mean_precision, mean_recall, interpretation, scores, justification}` |
| `calculate_bert_score` | `(generated, references, model_type)` | `{mean_f1, mean_precision, mean_recall, interpretation, model_used, scores, justification}` |
| `calculate_all_metrics` | `(questions, rag_results, generated, references)` | Combined dict with summary |
| `llm_judge_score` | `(model, question, generated, reference)` | `{accuracy, completeness, relevance, coherence}` (1-5 each) |

### 6c. Section 2.4: Evaluation Pipeline

**Script**: `ConversationalAI_Assignment_2_Group_122_Part2_Evaluation.py`

**Pipeline steps:**

1. Load `evaluation_questions.json` (100 Q&A pairs)
2. Load corpus (fixed_url.json + random_url.json -> 500 docs)
3. Load models (SentenceTransformer + TinyLlama)
4. Build indices (FAISS + BM25)
5. Run RAG on all 100 questions (hybrid, rrf_k=60)
6. Compute MRR (URL-level matching)
7. Compute ROUGE-L (answer overlap)
8. Compute BERTScore (semantic similarity)
9. Run ablation study (5 configurations)
10. Perform error analysis (retrieval failures, generation failures, category breakdown)
11. Save `evaluation_results.csv`
12. Save `ablation_results.json`
13. Generate `evaluation_report.html` with embedded charts

**Key functions:**

| Function | Purpose |
|----------|---------|
| `load_models()` | Load SentenceTransformer + TinyLlama |
| `load_corpus()` | Load 500 Wikipedia documents |
| `build_indices(corpus, model)` | Build FAISS index + BM25 index |
| `retrieve_dense(query, ...)` | FAISS similarity search |
| `retrieve_bm25(query, ...)` | BM25 keyword search |
| `retrieve_hybrid(query, ..., rrf_k)` | RRF fusion of dense + sparse |
| `build_context(results, max_chars)` | Concatenate top-N chunk contents (max 800 chars) |
| `generate_answer(query, context, llm)` | TinyLlama response generation |
| `run_rag_pipeline(questions, ..., method, rrf_k)` | Full RAG on all questions |
| `run_ablation_study(questions, ...)` | 5-config comparison |
| `perform_error_analysis(results, mrr_details)` | Categorize failures |
| `generate_html_report(results, metrics, ablation, errors, path)` | HTML report with embedded PNG charts |

**Ablation configurations:**

| Config | Method | RRF k | Description |
|--------|--------|-------|-------------|
| Dense-only | dense | - | FAISS semantic search only |
| Sparse-only | sparse | - | BM25 keyword search only |
| Hybrid (k=30) | hybrid | 30 | RRF with high top-rank emphasis |
| Hybrid (k=60) | hybrid | 60 | RRF default (balanced) |
| Hybrid (k=100) | hybrid | 100 | RRF with more uniform weighting |

### 6d. Evaluation Dashboard

**Script**: `ConversationalAI_Assignment_2_Group_122_Part2_Dashboard.py`

- **Framework**: Streamlit with Plotly charts
- **Run**: `streamlit run ConversationalAI_Assignment_2_Group_122_Part2_Dashboard.py` -> opens at http://localhost:8501

**5 Tabs:**

| Tab | Content | Charts |
|-----|---------|--------|
| Overview | Key metrics cards, question distribution, metric radar | Donut chart, Radar chart |
| Question Generation | Pipeline flow, browse/filter questions | Flow diagram |
| Evaluation Metrics | MRR/ROUGE-L/BERTScore sub-tabs with formulas | Gauge charts, histograms, box plots, scatter |
| Ablation Study | Method comparison, RRF parameter exploration | Grouped bars, line chart |
| Error Analysis | Failure categorization, example failures | Pie chart, bar chart |

- **Sidebar**: Data status indicators, Refresh/Recalculate buttons, HTML report export, paper links

---

## 7. Evaluation Report

> This section reproduces the evaluation report content in Markdown form. Run the evaluation pipeline to generate the full interactive HTML report at `evaluation_report.html`.
>
### 7a. Overall Performance Summary

**Metrics summary:**

| Metric | Score | Interpretation |
|--------|-------|----------------|
| MRR | 0.9733 | Excellent |
| ROUGE-L F1 | 0.3295 | Moderate lexical overlap |
| BERTScore F1 | 0.3814 | Low-Moderate semantic similarity |

**Retrieval statistics:**

| Statistic | Value |
|-----------|-------|
| Total Questions | 100 |
| Source Found in Top-5 | 99/100 |
| Top-1 Accuracy | 96% |
| Top-3 Accuracy | 99% |

**Interpretation**: The hybrid RAG system achieves excellent retrieval quality (MRR = 0.9733), meaning the correct source document is found at rank 1 for 96% of questions. The moderate ROUGE-L score (0.3295) reflects TinyLlama's limited capacity for precise wording -- generated answers often use different phrasing than the ground truth. The BERTScore (0.3814) is similarly moderate, indicating that while retrieval is near-perfect, the generation quality is constrained by the small 1.1B-parameter model. The gap between retrieval performance (MRR 0.97) and generation quality (ROUGE-L 0.33) confirms that the bottleneck is in generation, not retrieval.

### 7b. Metric Justifications

#### Metric 1: ROUGE-L

**Why chosen**: ROUGE-L measures the longest common subsequence (LCS) between generated and reference answers. It captures:

1. **Word overlap**: Whether the generated answer uses the same vocabulary as the ground truth
2. **Answer completeness**: Via recall -- how much of the reference answer is covered
3. **Order preservation**: LCS inherently respects word order, unlike bag-of-words metrics
4. **Stemming support**: Uses Porter stemmer to match morphological variants (e.g., "running" matches "runs")
5. **Standard metric**: Widely used in summarization and text generation evaluation (Lin, 2004)

**Calculation methodology:**

```
P_lcs = LCS(generated, reference) / len(generated)    -- Precision
R_lcs = LCS(generated, reference) / len(reference)    -- Recall
ROUGE-L = F1 = 2 * P_lcs * R_lcs / (P_lcs + R_lcs)  -- F-measure
```

Implementation uses `rouge_score.RougeScorer(['rougeL'], use_stemmer=True)`.

**Interpretation guidelines:**

| ROUGE-L F1 | Meaning |
|------------|---------|
| > 0.5 | High lexical overlap -- generated answer closely matches reference wording |
| 0.3-0.5 | Moderate overlap -- partially correct with different phrasing |
| < 0.3 | Low overlap -- answer may be correct but uses very different wording, or is wrong |

**Limitation**: ROUGE-L cannot detect semantic equivalence (e.g., "car" vs "automobile" scores 0). This is why BERTScore is used as the second metric.

#### Metric 2: BERTScore

**Why chosen**: BERTScore computes token-level cosine similarity using contextual BERT embeddings. It captures:

1. **Semantic equivalence**: "automobile" and "car", "purchase" and "buy" are recognized as similar
2. **Paraphrase detection**: Different sentence structures with the same meaning score high
3. **Human correlation**: Research shows BERTScore correlates with human judgment better than n-gram metrics (Zhang et al., 2020)
4. **Contextual understanding**: BERT embeddings capture word meaning in context (e.g., "bank" as financial vs river)
5. **Complementary to ROUGE-L**: Where ROUGE-L fails on paraphrases, BERTScore succeeds

**Calculation methodology:**

```
For each token t_i in generated answer:
  Find max cosine similarity with any token in reference: max_j cos(h_i, h_j)

P_BERT = (1/|gen|) * SUM_i max_j cos(h_gen_i, h_ref_j)   -- Precision
R_BERT = (1/|ref|) * SUM_j max_i cos(h_gen_i, h_ref_j)   -- Recall
F1_BERT = 2 * P_BERT * R_BERT / (P_BERT + R_BERT)        -- F-measure
```

Implementation uses `bert_score` library with model `distilbert-base-uncased`, `rescale_with_baseline=True`.

**Interpretation guidelines:**

| BERTScore F1 | Meaning |
|--------------|---------|
| > 0.7 | High semantic similarity -- answer conveys the same meaning |
| 0.5-0.7 | Moderate similarity -- partially correct meaning |
| < 0.5 | Low similarity -- answer likely incorrect or off-topic |

**Why both metrics together**: ROUGE-L catches exact and near-exact lexical matches; BERTScore catches semantically equivalent but differently worded answers. Together they cover both lexical and semantic quality dimensions, providing a more complete picture of answer quality than either metric alone.

### 7c. MRR (Mandatory Metric) Documentation

- **What it measures**: How well the retrieval stage finds the correct source document
- **Formula**: `MRR = (1/|Q|) * SUM (1/rank_i)` where rank_i is the position of the correct source URL in retrieved results
- **Computed at URL level**: Matches `question.source_url` against `result.retrieved_chunks[*].document.url`

**Interpretation:**

| MRR | Quality |
|-----|---------|
| > 0.8 | Excellent -- correct doc usually at rank 1 |
| 0.6-0.8 | Good -- correct doc usually in top 2-3 |
| 0.4-0.6 | Moderate -- correct doc sometimes in top 5 |
| < 0.4 | Poor -- retrieval frequently fails |

### 7d. Results Table

First 20 questions from `evaluation_results.csv`:

| Q ID | Category | Question (truncated) | RR | ROUGE-L | BERTScore |
|------|----------|---------------------|----|---------|-----------|
| q001 | factual | What is Faree known for? | 1.00 | 0.333 | 0.180 |
| q002 | factual | What is Mahmoud Kafil Uddin known for? | 1.00 | 0.615 | 0.786 |
| q003 | factual | Where is 9/11 Heroes Medal of Valor located? | 1.00 | 0.286 | 0.280 |
| q004 | factual | When was Lenny Mirra established/born? | 1.00 | 0.500 | 0.310 |
| q005 | factual | What is Molly Springfield's background...? | 1.00 | 0.559 | 0.590 |
| q006 | factual | What are The Globe Building...? | 1.00 | 0.586 | 0.626 |
| q007 | factual | What is the percentage of Earth's terrestrial...? | 0.50 | 0.000 | 0.189 |
| q008 | factual | What is the oldest and most prestigious...? | 1.00 | 0.129 | -0.034 |
| q009 | factual | Who is List of Superfund sites in Idaho? | 1.00 | 0.314 | 0.442 |
| q010 | factual | What is Greenpeace's goal...? | 0.33 | 0.176 | 0.308 |
| q011 | factual | Who is Spruce Run Evangelical Lutheran...? | 1.00 | 0.533 | 0.526 |
| q012 | factual | What is Social media measurement known for? | 1.00 | 0.233 | 0.477 |
| q013 | factual | Who is Magnus Eiriksson? | 1.00 | 0.437 | 0.531 |
| q014 | factual | What does Brazil refer to? | 1.00 | 0.424 | 0.443 |
| q015 | factual | Where is Lake Elizabeth...? | 1.00 | 0.500 | 0.505 |
| q016 | factual | What is Elon Musk's net worth...? | 1.00 | 0.348 | 0.433 |
| q017 | factual | What team represented the Maori community...? | 1.00 | 1.000 | 1.000 |
| q018 | factual | What was the impact of "Careful" on radio...? | 1.00 | 0.368 | 0.390 |
| q019 | factual | What was the purpose of the Act of 1813...? | 1.00 | 0.274 | 0.413 |
| q020 | factual | What does Big Bang refer to? | 1.00 | 0.933 | 0.973 |

See `evaluation_results.csv` for all 100 questions with full generated answers and ground truth.

### 7e. Visualizations

The `evaluation_report.html` contains the following embedded visualizations:

1. **Score Distributions** (3 histograms):
   - MRR reciprocal rank distribution -- shows how many questions get rank 1, 2, 3, etc.
   - ROUGE-L F1 distribution -- bell curve showing answer quality spread
   - BERTScore F1 distribution -- typically right-skewed (most answers have decent semantic similarity)

2. **Ablation Comparison** (grouped bar chart):
   - 5 methods on x-axis, 3 metric bars per method (MRR, ROUGE-L, BERTScore)
   - Demonstrates hybrid > dense-only > sparse-only

3. **Category Performance** (bar chart):
   - Factual vs Comparative vs Inferential vs Multi-hop
   - Shows which question types are hardest for the RAG system

4. **Metric Correlation** (scatter plot):
   - ROUGE-L (x-axis) vs BERTScore (y-axis), colored by category
   - Shows relationship between lexical and semantic quality

5. **Response Time Breakdown** (bar chart):
   - Retrieval time vs Generation time per question
   - Generation dominates (~10-20s vs ~0.1-0.5s retrieval)

Open `evaluation_report.html` in a browser to view all interactive visualizations.

### 7f. Ablation Study Results

| Method | MRR | ROUGE-L F1 | BERTScore F1 | Top-1 Acc | Top-3 Acc |
|--------|-----|------------|--------------|-----------|-----------|
| Dense-only | 0.9533 | 0.3386 | 0.3703 | 93% | 98% |
| Sparse-only | 0.9300 | 0.3173 | 0.3555 | 90% | 97% |
| Hybrid (k=30) | 0.9733 | 0.3334 | 0.3775 | 96% | 99% |
| **Hybrid (k=60)** | **0.9733** | **0.3332** | **0.3791** | **96%** | **99%** |
| Hybrid (k=100) | 0.9733 | 0.3234 | 0.3664 | 96% | 99% |

**Key findings:**

- Hybrid retrieval outperforms both single-method baselines in MRR and BERTScore
- Dense-only beats Sparse-only (semantic similarity is more valuable than keyword matching for this corpus)
- All three hybrid configurations (k=30, 60, 100) achieve the same MRR (0.9733), but k=60 has the highest BERTScore (0.3791)
- RRF k=60 is the optimal parameter -- it achieves the best balance of lexical and semantic generation quality
- Improvement over Dense-only: +2% MRR, +2% BERTScore
- Improvement over Sparse-only: +5% MRR, +5% ROUGE-L, +7% BERTScore
- The biggest gains from hybrid fusion are in retrieval precision: Top-1 accuracy jumps from 90-93% (single methods) to 96% (hybrid)

### 7g. LLM-as-Judge Results

The evaluation pipeline uses TinyLlama itself as an LLM judge, rating answers on 4 dimensions (1-5 scale):

| Dimension | Description | Avg Score |
|-----------|-------------|-----------|
| Accuracy | Is the information factually correct? | 3.5/5 |
| Completeness | Does it fully answer the question? | 3.0/5 |
| Relevance | Is the answer on-topic? | 4.0/5 |
| Coherence | Is it well-structured and readable? | 3.8/5 |

> Note: LLM-as-Judge scores are approximate as TinyLlama's own judgment capacity is limited. These scores should be interpreted as directional indicators rather than precise measurements.

### 7h. Error Analysis

**Failure categories:**

1. **Retrieval Failures** (1% of questions): Source document not found in top-5 results
   - Only 1 out of 100 questions had a reciprocal rank of 0.0 (q099: "What is the author's birthdate?" -- ambiguous query with no clear target article)
   - The hybrid retrieval system is highly effective, with 99/100 questions finding the source document

2. **Generation Failures** (~30% of questions): Correct context retrieved but ROUGE-L < 0.2 or BERTScore < 0.2
   - Common cause: TinyLlama's limited capacity (1.1B params), context truncation, hallucination
   - Example: Dates and numbers are sometimes hallucinated; model sometimes repeats the question or outputs "helpful assistant"

3. **Category breakdown:**

| Category | Count | MRR | ROUGE-L | BERTScore | Top-1 Acc | Source Found |
|----------|-------|-----|---------|-----------|-----------|-------------|
| Factual | 40 | 0.9708 | 0.3861 | 0.3869 | 95% | 100% |
| Comparative | 20 | 0.9750 | 0.2607 | 0.3191 | 95% | 100% |
| Inferential | 25 | 1.0000 | 0.3060 | 0.4068 | 100% | 100% |
| Multi-hop | 15 | 0.9333 | 0.3092 | 0.4074 | 93% | 93% |

**Key observations by category:**
- **Inferential** questions achieve perfect retrieval (MRR = 1.0) and the highest BERTScore (0.4068), suggesting the model handles reasoning questions well when the right context is provided
- **Factual** questions have the highest ROUGE-L (0.3861) due to more direct, specific answers that overlap lexically with ground truth
- **Comparative** questions have the lowest ROUGE-L (0.2607), as comparisons require synthesizing information that is often phrased differently from the reference
- **Multi-hop** questions have the lowest MRR (0.9333) and are the only category where retrieval sometimes fails entirely

**Example failures:**

1. **Retrieval failure (q099)**: "What is the author's birthdate?" -- Ambiguous query with no specific entity; the system retrieved William Shakespeare's article but the ground truth was "[author's birthdate]", a malformed question from the generation pipeline.

2. **Generation failure (q001)**: "What is Faree known for?" -- Despite correct retrieval (RR=1.0), TinyLlama generated "Faree is a helpful assistant" instead of information about the Maldivian film, showing the model's tendency to fall back to its instruction-tuning default.

3. **Generation failure (q008)**: Question about Tour de France Grand Tours -- TinyLlama hallucinated "Gironde d'Italie" instead of "Giro d'Italia", demonstrating the model's tendency to produce plausible-sounding but incorrect proper nouns (BERTScore = -0.034).

4. **Low-quality generation (q046)**: "How does Muhammad Ali compare to others?" -- Generated a reasonable-sounding answer but with minimal overlap with the ground truth (ROUGE-L = 0.051, BERTScore = -0.180), as the model produced generic praise rather than specific biographical facts.

---

## 8. Data Files

### 8a. Fixed URLs (200) -- Full List

The fixed dataset contains 200 curated Wikipedia articles across 8 domains:

**Domain distribution:**

| Domain | Count | Examples |
|--------|-------|----------|
| People | 50 | Einstein, Curie, Newton, Darwin, Tesla, Hawking, ... |
| Places | 30 | Grand Canyon, Everest, Tokyo, London, Paris, ... |
| Events | 25 | WWII, French Revolution, Moon landing, Cold War, ... |
| Technology | 25 | AI, Internet, Blockchain, Bitcoin, Machine learning, ... |
| Science | 25 | Climate change, DNA, Black hole, Quantum mechanics, ... |
| Arts | 20 | Mona Lisa, Starry Night, Taj Mahal, Colosseum, ... |
| Sports | 15 | Olympics, FIFA World Cup, NBA, Formula One, ... |
| Organization | 10 | UN, NASA, WHO, NATO, Red Cross, ... |
| **Total** | **200** | |

<details>
<summary>Click to expand: Complete list of 200 fixed URLs</summary>

```json
[
  // PEOPLE (50)
  {"url": "https://en.wikipedia.org/wiki/Albert_Einstein", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Marie_Curie", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Isaac_Newton", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Charles_Darwin", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Nikola_Tesla", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Stephen_Hawking", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Galileo_Galilei", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Richard_Feynman", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Ada_Lovelace", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Alan_Turing", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Abraham_Lincoln", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Winston_Churchill", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Mahatma_Gandhi", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Nelson_Mandela", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Martin_Luther_King_Jr.", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Napoleon", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Cleopatra", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Julius_Caesar", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Queen_Victoria", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Alexander_the_Great", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Leonardo_da_Vinci", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Vincent_van_Gogh", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Pablo_Picasso", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Michelangelo", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/William_Shakespeare", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Wolfgang_Amadeus_Mozart", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Ludwig_van_Beethoven", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Johann_Sebastian_Bach", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Frida_Kahlo", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Claude_Monet", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Elon_Musk", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Steve_Jobs", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Bill_Gates", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Jeff_Bezos", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Mark_Zuckerberg", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Oprah_Winfrey", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Barack_Obama", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Angela_Merkel", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Malala_Yousafzai", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Greta_Thunberg", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Michael_Jordan", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Lionel_Messi", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Cristiano_Ronaldo", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Serena_Williams", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Usain_Bolt", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Muhammad_Ali", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Roger_Federer", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/LeBron_James", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Tiger_Woods", "domain": "people"},
  {"url": "https://en.wikipedia.org/wiki/Michael_Phelps", "domain": "people"},

  // PLACES (30)
  {"url": "https://en.wikipedia.org/wiki/Grand_Canyon", "domain": "places"},
  {"url": "https://en.wikipedia.org/wiki/Mount_Everest", "domain": "places"},
  {"url": "https://en.wikipedia.org/wiki/Niagara_Falls", "domain": "places"},
  {"url": "https://en.wikipedia.org/wiki/Great_Barrier_Reef", "domain": "places"},
  {"url": "https://en.wikipedia.org/wiki/Amazon_rainforest", "domain": "places"},
  {"url": "https://en.wikipedia.org/wiki/Sahara", "domain": "places"},
  {"url": "https://en.wikipedia.org/wiki/Victoria_Falls", "domain": "places"},
  {"url": "https://en.wikipedia.org/wiki/Yellowstone_National_Park", "domain": "places"},
  {"url": "https://en.wikipedia.org/wiki/Tokyo", "domain": "places"},
  {"url": "https://en.wikipedia.org/wiki/New_York_City", "domain": "places"},
  {"url": "https://en.wikipedia.org/wiki/London", "domain": "places"},
  {"url": "https://en.wikipedia.org/wiki/Paris", "domain": "places"},
  {"url": "https://en.wikipedia.org/wiki/Rome", "domain": "places"},
  {"url": "https://en.wikipedia.org/wiki/Sydney", "domain": "places"},
  {"url": "https://en.wikipedia.org/wiki/Dubai", "domain": "places"},
  {"url": "https://en.wikipedia.org/wiki/Singapore", "domain": "places"},
  {"url": "https://en.wikipedia.org/wiki/Hong_Kong", "domain": "places"},
  {"url": "https://en.wikipedia.org/wiki/Berlin", "domain": "places"},
  {"url": "https://en.wikipedia.org/wiki/United_States", "domain": "places"},
  {"url": "https://en.wikipedia.org/wiki/China", "domain": "places"},
  {"url": "https://en.wikipedia.org/wiki/India", "domain": "places"},
  {"url": "https://en.wikipedia.org/wiki/Japan", "domain": "places"},
  {"url": "https://en.wikipedia.org/wiki/Brazil", "domain": "places"},
  {"url": "https://en.wikipedia.org/wiki/Australia", "domain": "places"},
  {"url": "https://en.wikipedia.org/wiki/Canada", "domain": "places"},
  {"url": "https://en.wikipedia.org/wiki/Germany", "domain": "places"},
  {"url": "https://en.wikipedia.org/wiki/France", "domain": "places"},
  {"url": "https://en.wikipedia.org/wiki/United_Kingdom", "domain": "places"},
  {"url": "https://en.wikipedia.org/wiki/Italy", "domain": "places"},
  {"url": "https://en.wikipedia.org/wiki/Russia", "domain": "places"},

  // EVENTS (25)
  {"url": "https://en.wikipedia.org/wiki/World_War_II", "domain": "events"},
  {"url": "https://en.wikipedia.org/wiki/World_War_I", "domain": "events"},
  {"url": "https://en.wikipedia.org/wiki/French_Revolution", "domain": "events"},
  {"url": "https://en.wikipedia.org/wiki/American_Revolution", "domain": "events"},
  {"url": "https://en.wikipedia.org/wiki/Industrial_Revolution", "domain": "events"},
  {"url": "https://en.wikipedia.org/wiki/Renaissance", "domain": "events"},
  {"url": "https://en.wikipedia.org/wiki/Cold_War", "domain": "events"},
  {"url": "https://en.wikipedia.org/wiki/Moon_landing", "domain": "events"},
  {"url": "https://en.wikipedia.org/wiki/Fall_of_the_Berlin_Wall", "domain": "events"},
  {"url": "https://en.wikipedia.org/wiki/September_11_attacks", "domain": "events"},
  {"url": "https://en.wikipedia.org/wiki/Black_Death", "domain": "events"},
  {"url": "https://en.wikipedia.org/wiki/Reformation", "domain": "events"},
  {"url": "https://en.wikipedia.org/wiki/Russian_Revolution", "domain": "events"},
  {"url": "https://en.wikipedia.org/wiki/American_Civil_War", "domain": "events"},
  {"url": "https://en.wikipedia.org/wiki/Hiroshima_and_Nagasaki_atomic_bombings", "domain": "events"},
  {"url": "https://en.wikipedia.org/wiki/COVID-19_pandemic", "domain": "events"},
  {"url": "https://en.wikipedia.org/wiki/Great_Depression", "domain": "events"},
  {"url": "https://en.wikipedia.org/wiki/Chernobyl_disaster", "domain": "events"},
  {"url": "https://en.wikipedia.org/wiki/Titanic", "domain": "events"},
  {"url": "https://en.wikipedia.org/wiki/Discovery_of_the_Americas", "domain": "events"},
  {"url": "https://en.wikipedia.org/wiki/Ancient_Egypt", "domain": "events"},
  {"url": "https://en.wikipedia.org/wiki/Roman_Empire", "domain": "events"},
  {"url": "https://en.wikipedia.org/wiki/Ancient_Greece", "domain": "events"},
  {"url": "https://en.wikipedia.org/wiki/Byzantine_Empire", "domain": "events"},
  {"url": "https://en.wikipedia.org/wiki/Ottoman_Empire", "domain": "events"},

  // TECHNOLOGY (25)
  {"url": "https://en.wikipedia.org/wiki/Artificial_intelligence", "domain": "technology"},
  {"url": "https://en.wikipedia.org/wiki/Internet", "domain": "technology"},
  {"url": "https://en.wikipedia.org/wiki/Computer", "domain": "technology"},
  {"url": "https://en.wikipedia.org/wiki/Smartphone", "domain": "technology"},
  {"url": "https://en.wikipedia.org/wiki/World_Wide_Web", "domain": "technology"},
  {"url": "https://en.wikipedia.org/wiki/Machine_learning", "domain": "technology"},
  {"url": "https://en.wikipedia.org/wiki/Blockchain", "domain": "technology"},
  {"url": "https://en.wikipedia.org/wiki/Cryptocurrency", "domain": "technology"},
  {"url": "https://en.wikipedia.org/wiki/Bitcoin", "domain": "technology"},
  {"url": "https://en.wikipedia.org/wiki/Electric_vehicle", "domain": "technology"},
  {"url": "https://en.wikipedia.org/wiki/Robotics", "domain": "technology"},
  {"url": "https://en.wikipedia.org/wiki/3D_printing", "domain": "technology"},
  {"url": "https://en.wikipedia.org/wiki/Virtual_reality", "domain": "technology"},
  {"url": "https://en.wikipedia.org/wiki/Augmented_reality", "domain": "technology"},
  {"url": "https://en.wikipedia.org/wiki/Cloud_computing", "domain": "technology"},
  {"url": "https://en.wikipedia.org/wiki/5G", "domain": "technology"},
  {"url": "https://en.wikipedia.org/wiki/Quantum_computing", "domain": "technology"},
  {"url": "https://en.wikipedia.org/wiki/Self-driving_car", "domain": "technology"},
  {"url": "https://en.wikipedia.org/wiki/SpaceX", "domain": "technology"},
  {"url": "https://en.wikipedia.org/wiki/Tesla,_Inc.", "domain": "technology"},
  {"url": "https://en.wikipedia.org/wiki/Apple_Inc.", "domain": "technology"},
  {"url": "https://en.wikipedia.org/wiki/Google", "domain": "technology"},
  {"url": "https://en.wikipedia.org/wiki/Microsoft", "domain": "technology"},
  {"url": "https://en.wikipedia.org/wiki/Amazon_(company)", "domain": "technology"},
  {"url": "https://en.wikipedia.org/wiki/Facebook", "domain": "technology"},

  // SCIENCE (25)
  {"url": "https://en.wikipedia.org/wiki/Climate_change", "domain": "science"},
  {"url": "https://en.wikipedia.org/wiki/DNA", "domain": "science"},
  {"url": "https://en.wikipedia.org/wiki/Evolution", "domain": "science"},
  {"url": "https://en.wikipedia.org/wiki/Black_hole", "domain": "science"},
  {"url": "https://en.wikipedia.org/wiki/Big_Bang", "domain": "science"},
  {"url": "https://en.wikipedia.org/wiki/Quantum_mechanics", "domain": "science"},
  {"url": "https://en.wikipedia.org/wiki/Theory_of_relativity", "domain": "science"},
  {"url": "https://en.wikipedia.org/wiki/Photosynthesis", "domain": "science"},
  {"url": "https://en.wikipedia.org/wiki/Human_brain", "domain": "science"},
  {"url": "https://en.wikipedia.org/wiki/Solar_System", "domain": "science"},
  {"url": "https://en.wikipedia.org/wiki/Milky_Way", "domain": "science"},
  {"url": "https://en.wikipedia.org/wiki/Periodic_table", "domain": "science"},
  {"url": "https://en.wikipedia.org/wiki/Atom", "domain": "science"},
  {"url": "https://en.wikipedia.org/wiki/Cell_(biology)", "domain": "science"},
  {"url": "https://en.wikipedia.org/wiki/Genetics", "domain": "science"},
  {"url": "https://en.wikipedia.org/wiki/Vaccination", "domain": "science"},
  {"url": "https://en.wikipedia.org/wiki/Antibiotic", "domain": "science"},
  {"url": "https://en.wikipedia.org/wiki/Nuclear_power", "domain": "science"},
  {"url": "https://en.wikipedia.org/wiki/Renewable_energy", "domain": "science"},
  {"url": "https://en.wikipedia.org/wiki/Global_warming", "domain": "science"},
  {"url": "https://en.wikipedia.org/wiki/Biodiversity", "domain": "science"},
  {"url": "https://en.wikipedia.org/wiki/Ecosystem", "domain": "science"},
  {"url": "https://en.wikipedia.org/wiki/Oceanography", "domain": "science"},
  {"url": "https://en.wikipedia.org/wiki/Astronomy", "domain": "science"},
  {"url": "https://en.wikipedia.org/wiki/Neuroscience", "domain": "science"},

  // ARTS (20)
  {"url": "https://en.wikipedia.org/wiki/Mona_Lisa", "domain": "arts"},
  {"url": "https://en.wikipedia.org/wiki/Starry_Night", "domain": "arts"},
  {"url": "https://en.wikipedia.org/wiki/The_Last_Supper_(Leonardo)", "domain": "arts"},
  {"url": "https://en.wikipedia.org/wiki/Sistine_Chapel_ceiling", "domain": "arts"},
  {"url": "https://en.wikipedia.org/wiki/The_Scream", "domain": "arts"},
  {"url": "https://en.wikipedia.org/wiki/Girl_with_a_Pearl_Earring", "domain": "arts"},
  {"url": "https://en.wikipedia.org/wiki/The_Birth_of_Venus", "domain": "arts"},
  {"url": "https://en.wikipedia.org/wiki/Guernica_(Picasso)", "domain": "arts"},
  {"url": "https://en.wikipedia.org/wiki/The_Persistence_of_Memory", "domain": "arts"},
  {"url": "https://en.wikipedia.org/wiki/The_Great_Wave_off_Kanagawa", "domain": "arts"},
  {"url": "https://en.wikipedia.org/wiki/Statue_of_Liberty", "domain": "arts"},
  {"url": "https://en.wikipedia.org/wiki/Eiffel_Tower", "domain": "arts"},
  {"url": "https://en.wikipedia.org/wiki/Great_Wall_of_China", "domain": "arts"},
  {"url": "https://en.wikipedia.org/wiki/Taj_Mahal", "domain": "arts"},
  {"url": "https://en.wikipedia.org/wiki/Colosseum", "domain": "arts"},
  {"url": "https://en.wikipedia.org/wiki/Machu_Picchu", "domain": "arts"},
  {"url": "https://en.wikipedia.org/wiki/Petra", "domain": "arts"},
  {"url": "https://en.wikipedia.org/wiki/Louvre", "domain": "arts"},
  {"url": "https://en.wikipedia.org/wiki/Metropolitan_Museum_of_Art", "domain": "arts"},
  {"url": "https://en.wikipedia.org/wiki/British_Museum", "domain": "arts"},

  // SPORTS (15)
  {"url": "https://en.wikipedia.org/wiki/Olympic_Games", "domain": "sports"},
  {"url": "https://en.wikipedia.org/wiki/FIFA_World_Cup", "domain": "sports"},
  {"url": "https://en.wikipedia.org/wiki/Super_Bowl", "domain": "sports"},
  {"url": "https://en.wikipedia.org/wiki/NBA", "domain": "sports"},
  {"url": "https://en.wikipedia.org/wiki/UEFA_Champions_League", "domain": "sports"},
  {"url": "https://en.wikipedia.org/wiki/Wimbledon_Championships", "domain": "sports"},
  {"url": "https://en.wikipedia.org/wiki/Tour_de_France", "domain": "sports"},
  {"url": "https://en.wikipedia.org/wiki/Formula_One", "domain": "sports"},
  {"url": "https://en.wikipedia.org/wiki/Cricket_World_Cup", "domain": "sports"},
  {"url": "https://en.wikipedia.org/wiki/Rugby_World_Cup", "domain": "sports"},
  {"url": "https://en.wikipedia.org/wiki/Association_football", "domain": "sports"},
  {"url": "https://en.wikipedia.org/wiki/Basketball", "domain": "sports"},
  {"url": "https://en.wikipedia.org/wiki/Tennis", "domain": "sports"},
  {"url": "https://en.wikipedia.org/wiki/Golf", "domain": "sports"},
  {"url": "https://en.wikipedia.org/wiki/Swimming_(sport)", "domain": "sports"},

  // ORGANIZATION (10)
  {"url": "https://en.wikipedia.org/wiki/United_Nations", "domain": "organization"},
  {"url": "https://en.wikipedia.org/wiki/World_Health_Organization", "domain": "organization"},
  {"url": "https://en.wikipedia.org/wiki/NASA", "domain": "organization"},
  {"url": "https://en.wikipedia.org/wiki/European_Union", "domain": "organization"},
  {"url": "https://en.wikipedia.org/wiki/NATO", "domain": "organization"},
  {"url": "https://en.wikipedia.org/wiki/Red_Cross", "domain": "organization"},
  {"url": "https://en.wikipedia.org/wiki/Greenpeace", "domain": "organization"},
  {"url": "https://en.wikipedia.org/wiki/Amnesty_International", "domain": "organization"},
  {"url": "https://en.wikipedia.org/wiki/World_Bank", "domain": "organization"},
  {"url": "https://en.wikipedia.org/wiki/International_Monetary_Fund", "domain": "organization"}
]
```

</details>

### 8b. Random URLs (300)

- Generated at runtime via MediaWiki API `list=random`
- Filtered for minimum 200 words per article
- Domain classified using category keyword matching
- Different set each run (timestamp-based seed)
- Stored in `random_url.json`
- Excludes any URLs already in the fixed set

### 8c. Evaluation Questions (100)

- **Schema**: `{id, question, ground_truth, source_url, source_id, category, domain}`
- **Distribution**: 40 factual, 20 comparative, 25 inferential, 15 multi-hop
- **Stored in**: `evaluation_questions.json`
- **Regenerate**: `python ConversationalAI_Assignment_2_Group_122_Part2_QuestionGen.py`

**Actual domain distribution** (from current `evaluation_questions.json`):

| Domain | Count |
|--------|-------|
| people | 32 |
| places | 21 |
| other | 13 |
| arts | 10 |
| science | 7 |
| sports | 7 |
| events | 6 |
| technology | 3 |
| organization | 1 |
| **Total** | **100** |

### 8d. JSON Schema Reference

```json
// fixed_url.json / random_url.json entry
{
  "id": "a79f6efb-ca0d-46a4-9a87-749c3e9084be",
  "url": "https://en.wikipedia.org/wiki/Albert_Einstein",
  "domain": "people",
  "content": "Albert Einstein (14 March 1879 - 18 April 1955) was a German-born..."
}

// evaluation_questions.json entry
{
  "id": "q001",
  "question": "What is Albert Einstein known for?",
  "ground_truth": "Albert Einstein was a German-born theoretical physicist...",
  "source_url": "https://en.wikipedia.org/wiki/Albert_Einstein",
  "source_id": "a79f6efb-ca0d-46a4-9a87-749c3e9084be",
  "category": "factual",
  "domain": "people"
}
```

### 8e. Output Files

| File | Generated By | Contents | Regenerate Command |
|------|-------------|----------|-------------------|
| `evaluation_results.csv` | Part2_Evaluation.py | Per-question: ID, category, question, ground_truth, generated_answer, source_url, reciprocal_rank, rouge_l_f1, bert_score_f1 | `python ConversationalAI_Assignment_2_Group_122_Part2_Evaluation.py` |
| `ablation_results.json` | Part2_Evaluation.py | 5-config comparison: method, mrr, rouge_l_f1, bert_score_f1, top1/top3 accuracy | Same |
| `evaluation_report.html` | Part2_Evaluation.py | Full HTML report with embedded PNG charts, tables, error analysis | Same |

---

## 9. Technical Notes

### 9a. Architecture Decisions

- **Why TinyLlama?** Only 1.1B parameters, runs on CPU, 4-bit quantization keeps RAM under 2GB. Trade-off: lower quality answers vs zero GPU requirement.
- **Why FAISS IndexFlatIP?** Exact search (no approximation). 500 documents is small enough that brute-force inner product is fast (<50ms). No need for IVF/HNSW approximate indices.
- **Why RRF over other fusion methods?** RRF is parameter-light (single k), rank-based (no score normalization needed), and proven effective (Cormack et al., 2009). Alternatives like CombSUM require score calibration between different retrieval methods.
- **Why all-MiniLM-L6-v2?** Best speed/quality trade-off for sentence embeddings. 384 dimensions keeps FAISS index small. Only 22M parameters -- loads fast on CPU.

### 9b. Limitations

- **TinyLlama quality**: 1.1B parameter model frequently produces incomplete or slightly inaccurate answers. A 7B+ model would improve generation quality significantly.
- **Context window**: 2048 tokens limits how much retrieved content can be included. Only top-2 chunks fit with the prompt.
- **200-word extraction**: Articles are truncated to ~200 words, losing important content from longer articles.
- **Single-turn only**: No conversation history or follow-up question support.
- **Domain classification**: Keyword-based, not ML-based. Some articles are misclassified.
- **No chunk-level retrieval**: Entire documents (up to 200 words) are retrieved, not smaller chunks. With longer articles, chunking would improve precision.

### 9c. Reproduction Instructions

1. Clone the repository
2. Create venv and install requirements.txt
3. Run `python generate_urls.py --fixed 200 --random 300 --group GROUP_122` (or use existing JSONs)
4. Run `python ConversationalAI_Assignment_2_Group_122_Part2_QuestionGen.py`
5. Run `python ConversationalAI_Assignment_2_Group_122_Part2_Evaluation.py`
6. Open `evaluation_report.html` in a browser
7. Or run `.\run_part2.ps1` for the full pipeline

**Note on reproducibility**:
- Fixed URLs are deterministic (hardcoded list)
- Random URLs differ each run (timestamp seed)
- Question generation involves LLM randomness (temperature=0.7) + fallback templates
- Evaluation metrics are deterministic given the same questions and corpus

### 9d. References

1. **RRF**: Cormack, G. V., Clarke, C. L. A., & Buttcher, S. (2009). Reciprocal Rank Fusion outperforms Condorcet and Individual Rank Learning Methods. *SIGIR 2009*.
2. **BM25**: Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*.
3. **ROUGE**: Lin, C.-Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries. *Text Summarization Branches Out*.
4. **BERTScore**: Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). BERTScore: Evaluating Text Generation with BERT. *ICLR 2020*.
5. **MRR**: Voorhees, E. M. (1999). The TREC-8 Question Answering Track Report. *TREC 1999*.
6. **Sentence-BERT**: Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP 2019*.
7. **TinyLlama**: Zhang, P., et al. (2024). TinyLlama: An Open-Source Small Language Model. *arXiv:2401.02385*.

---

## 10. Challenges and Learning Experience

> This section documents the real-world challenges encountered and valuable lessons learned during the implementation of this Hybrid RAG System assignment.

### 10a. Environment Setup Challenges

| Challenge | Description | Resolution |
|-----------|-------------|------------|
| **ctransformers installation** | Initial `ModuleNotFoundError: No module named 'ctransformers'` when loading TinyLlama | Installed via `pip install ctransformers` -- learned that GGUF model loading requires specific library not included in standard transformers |
| **Hugging Face rate limiting** | Frequent "unauthenticated requests" warnings slowing model downloads | Set `HF_TOKEN` environment variable permanently using `[System.Environment]::SetEnvironmentVariable()` |
| **NLTK data missing** | `LookupError: punkt` tokenizer not found during BM25 preprocessing | Downloaded required NLTK data: `nltk.download('punkt')`, `nltk.download('stopwords')`, `nltk.download('punkt_tab')` |
| **PowerShell execution policy** | Script execution blocked with "running scripts is disabled on this system" | Resolved with `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| **SSL certificate errors** | Wikipedia API requests failing with SSL verification errors | Updated certifi package and set proper SSL context |

### 10b. Model-Related Challenges

| Challenge | Description | Learning |
|-----------|-------------|----------|
| **TinyLlama output quality** | Generated answers often incomplete, repetitive, or slightly inaccurate | Trade-off between model size (1.1B params) and quality -- larger models (7B+) would significantly improve generation but require GPU |
| **Context window limitations** | Only ~2048 tokens available, limiting how much retrieved content can be included | Learned to optimize context by truncating to 600 characters and using only top-2 chunks |
| **Slow inference on CPU** | Each question takes 10-20 seconds for generation | 4-bit quantization (Q4_K_M) helps but CPU inference is inherently slow -- batch processing and caching strategies are essential |
| **BERTScore model loading** | `embeddings.position_ids UNEXPECTED` warning on every load | Harmless warning from model architecture evolution -- can be safely ignored |
| **Model hallucination** | TinyLlama sometimes generates dates/numbers not present in context | Learned importance of prompt engineering -- explicit instructions to use "ONLY the provided context" reduce hallucination |

### 10c. Evaluation Pipeline Challenges

| Challenge | Description | Resolution |
|-----------|-------------|------------|
| **MRR calculation complexity** | Initially confused URL-level vs chunk-level matching | MRR must match `source_url` from question against retrieved document URLs, not individual chunks |
| **ROUGE-L interpretation** | Low ROUGE-L scores (~0.3-0.4) despite seemingly correct answers | Learned that ROUGE-L is lexical -- different wording of correct answers scores low. This justified adding BERTScore for semantic evaluation |
| **BERTScore computation time** | BERTScore much slower than ROUGE-L due to embedding computation | Batching references/candidates and using `distilbert-base-uncased` instead of larger BERT variants |
| **Question generation validation** | Many LLM-generated questions were malformed or unanswerable | Implemented fallback templates and validation steps -- learned that LLM outputs need post-processing |
| **Ablation study design** | Determining which configurations to compare | Selected 5 configurations covering single methods + hybrid with different k values to show RRF parameter sensitivity |

### 10d. Integration Challenges

| Challenge | Description | Learning |
|-----------|-------------|----------|
| **Pipeline orchestration** | Coordinating multiple Python scripts with dependencies | Created PowerShell scripts (`run_part1.ps1`, `run_part2.ps1`) with proper error handling, progress tracking, and skip options |
| **Data file dependencies** | Pipeline fails if intermediate files missing | Added file existence checks and clear error messages at each pipeline step |
| **Streamlit state management** | Dashboard losing state on refresh | Used `@st.cache_resource` and `@st.cache_data` decorators to persist expensive computations |
| **HTML report generation** | Embedding matplotlib charts in standalone HTML | Learned Base64 encoding of PNG images to create self-contained HTML reports |

### 10e. Key Learnings

#### Technical Learnings

1. **Hybrid Retrieval > Single Methods**: RRF fusion consistently outperformed both dense-only and sparse-only retrieval. The complementary nature of semantic (dense) and keyword (sparse) matching provides robustness.

2. **Metric Selection Matters**: ROUGE-L alone is insufficient -- semantically equivalent answers with different wording score poorly. BERTScore captures semantic similarity but is computationally expensive. Using both provides a complete picture.

3. **Small LLMs Have Trade-offs**: TinyLlama (1.1B params) enables CPU-only deployment but sacrifices answer quality. For production systems, larger models or API-based LLMs (GPT-4, Claude) would be necessary.

4. **Prompt Engineering is Critical**: The exact wording of prompts significantly affects output quality. Explicit instructions ("use ONLY the provided context") and structured formats (ChatML) help constrain the model.

5. **Reproducibility Requires Planning**: Hash-based seeding (`hash("GROUP_122")`) ensures deterministic URL sampling. However, LLM generation introduces randomness -- temperature=0.1 reduces but doesn't eliminate variation.

#### Process Learnings

1. **Iterative Development**: Started with simple implementations, then added complexity. For example, question generation evolved from pure LLM to LLM + fallback templates.

2. **Error Handling is Essential**: Production-quality code requires extensive try/except blocks, retries with backoff, and graceful degradation.

3. **Documentation During Development**: Writing CLAUDE.md alongside coding helped clarify implementation decisions and ensured nothing was forgotten.

4. **Visualization Aids Understanding**: The Streamlit dashboard made it much easier to understand metric distributions and identify failure patterns than raw CSV files.

### 10f. What I Would Do Differently

| Aspect | Original Approach | Better Approach |
|--------|------------------|-----------------|
| **Model choice** | TinyLlama for everything | Use TinyLlama for dev/testing, larger model (Llama-2-7B) for final evaluation |
| **Chunking strategy** | Whole documents (200 words) | Smaller overlapping chunks (100 words, 50-word overlap) for better retrieval precision |
| **Question generation** | Random document sampling | Stratified sampling by domain to ensure balanced coverage |
| **Evaluation set** | 100 questions | 200+ questions for more statistically significant results |
| **Error analysis** | Post-hoc categorization | Pre-defined failure taxonomy with automated classification |

### 10g. Time Investment

| Phase | Estimated Hours | Actual Hours | Notes |
|-------|-----------------|--------------|-------|
| Environment setup | 2 | 4 | Dependency conflicts, model downloads |
| Part 1 implementation | 8 | 12 | Wikipedia API rate limits, FAISS debugging |
| Part 2 question gen | 4 | 6 | LLM output validation |
| Part 2 metrics | 4 | 5 | ROUGE/BERTScore library quirks |
| Part 2 evaluation | 6 | 10 | Ablation study, error analysis |
| Dashboard | 4 | 6 | Plotly learning curve, Streamlit caching |
| Documentation | 4 | 5 | README, CLAUDE.md |
| **Total** | **32** | **48** | ~50% over initial estimate |

### 10h. Skills Developed

- **Information Retrieval**: Dense (FAISS) + Sparse (BM25) retrieval, Reciprocal Rank Fusion
- **NLP Metrics**: MRR, ROUGE-L, BERTScore implementation and interpretation
- **LLM Deployment**: ctransformers, GGUF quantization, prompt engineering
- **Evaluation Design**: Ablation studies, LLM-as-Judge, error analysis taxonomy
- **Full-Stack ML**: End-to-end pipeline from data collection to interactive dashboard
- **DevOps**: PowerShell automation, environment management, reproducibility practices
