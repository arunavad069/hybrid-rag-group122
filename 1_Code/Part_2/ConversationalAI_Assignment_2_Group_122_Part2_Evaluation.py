"""
Conversational AI Assignment 2 - Part 2: Automated Evaluation Pipeline
=======================================================================
Complete evaluation pipeline for the Hybrid RAG system.

Pipeline Steps:
1. Load evaluation questions (100 Q&A pairs)
2. Run RAG on each question
3. Compute metrics (MRR, ROUGE-L, BERTScore)
4. Run ablation study (dense-only, sparse-only, hybrid variants)
5. Run LLM-as-Judge evaluation
6. Perform error analysis
7. Generate outputs (CSV, JSON, HTML report)

Usage:
    python ConversationalAI_Assignment_2_Group_122_Part2_Evaluation.py

Outputs:
    - evaluation_results.csv: Per-question metrics
    - ablation_results.json: Ablation study comparisons
    - evaluation_report.html: Full HTML report with visualizations
"""

import json
import os
import time
import string
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import metrics module
from ConversationalAI_Assignment_2_Group_122_Part2_Metrics import (
    calculate_mrr,
    calculate_rouge_l,
    calculate_bert_score,
    llm_judge_score
)


# ============================================================================
# CONFIGURATION
# ============================================================================

QUESTIONS_FILE = "evaluation_questions.json"
FIXED_URL_FILE = "fixed_url.json"
RANDOM_URL_FILE = "random_url.json"

OUTPUT_CSV = "evaluation_results.csv"
OUTPUT_ABLATION = "ablation_results.json"
OUTPUT_REPORT = "evaluation_report.html"

# LLM Configuration
LLM_MODEL_REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
LLM_MODEL_FILE = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MAX_CONTEXT_LENGTH = 2048


# ============================================================================
# MODEL AND INDEX LOADING
# ============================================================================

def load_models():
    """Load all required models."""
    import certifi
    os.environ['SSL_CERT_FILE'] = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    os.environ['CURL_CA_BUNDLE'] = certifi.where()

    from sentence_transformers import SentenceTransformer
    from ctransformers import AutoModelForCausalLM

    print("Loading sentence transformer...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Loading TinyLlama...")
    CPU_THREADS = os.cpu_count() or 4
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_REPO,
        model_file=LLM_MODEL_FILE,
        model_type="llama",
        context_length=MAX_CONTEXT_LENGTH,
        gpu_layers=0,
        threads=CPU_THREADS,
        batch_size=512
    )

    print("Models loaded successfully!")
    return embedding_model, llm_model


def load_corpus() -> List[Dict]:
    """Load document corpus."""
    all_documents = []

    if os.path.exists(FIXED_URL_FILE):
        with open(FIXED_URL_FILE, 'r', encoding='utf-8') as f:
            all_documents.extend(json.load(f))

    if os.path.exists(RANDOM_URL_FILE):
        with open(RANDOM_URL_FILE, 'r', encoding='utf-8') as f:
            all_documents.extend(json.load(f))

    print(f"Loaded {len(all_documents)} documents")
    return all_documents


def build_indices(corpus: List[Dict], embedding_model):
    """Build FAISS and BM25 indices."""
    import faiss
    from rank_bm25 import BM25Okapi
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    # Build FAISS index
    print("Building FAISS index...")
    texts = [doc['content'] for doc in corpus]
    embeddings = embedding_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)

    embedding_dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(embedding_dim)
    faiss_index.add(embeddings)

    # Build BM25 index
    print("Building BM25 index...")
    stop_words = set(stopwords.words('english'))

    def preprocess(text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
        return tokens

    tokenized_corpus = [preprocess(doc['content']) for doc in corpus]
    bm25_index = BM25Okapi(tokenized_corpus)

    print("Indices built successfully!")
    return faiss_index, embeddings, bm25_index, preprocess


# ============================================================================
# RETRIEVAL FUNCTIONS
# ============================================================================

def retrieve_dense(query, model, index, documents, k=10):
    """Dense vector retrieval using FAISS."""
    import faiss

    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    similarities, indices = index.search(query_embedding, k)

    results = []
    for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
        if idx < len(documents):
            results.append({
                'rank': i + 1,
                'similarity_score': float(sim),
                'document': documents[idx]
            })
    return results


def retrieve_bm25(query, bm25_index, documents, preprocess_func, k=10):
    """BM25 sparse retrieval."""
    query_tokens = preprocess_func(query)
    scores = bm25_index.get_scores(query_tokens)

    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    results = []
    for rank, idx in enumerate(top_k_indices, 1):
        results.append({
            'rank': rank,
            'bm25_score': float(scores[idx]),
            'document': documents[idx]
        })
    return results


def retrieve_hybrid(query, embedding_model, faiss_index, bm25_index,
                    corpus, preprocess_func, top_k=10, top_n=5, rrf_k=60):
    """Hybrid retrieval using RRF."""
    dense_results = retrieve_dense(query, embedding_model, faiss_index, corpus, k=top_k)
    bm25_results = retrieve_bm25(query, bm25_index, corpus, preprocess_func, k=top_k)

    rrf_scores = {}
    doc_info = {}

    for result in dense_results:
        doc = result['document']
        doc_idx = corpus.index(doc) if doc in corpus else None

        if doc_idx is not None:
            rank = result['rank']
            if doc_idx not in rrf_scores:
                rrf_scores[doc_idx] = 0.0
                doc_info[doc_idx] = {'document': doc, 'dense_rank': None, 'bm25_rank': None}
            rrf_scores[doc_idx] += 1.0 / (rrf_k + rank)
            doc_info[doc_idx]['dense_rank'] = rank

    for result in bm25_results:
        doc = result['document']
        doc_idx = corpus.index(doc) if doc in corpus else None

        if doc_idx is not None:
            rank = result['rank']
            if doc_idx not in rrf_scores:
                rrf_scores[doc_idx] = 0.0
                doc_info[doc_idx] = {'document': doc, 'dense_rank': None, 'bm25_rank': None}
            rrf_scores[doc_idx] += 1.0 / (rrf_k + rank)
            doc_info[doc_idx]['bm25_rank'] = rank

    sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_n]

    final_results = []
    for rank, doc_idx in enumerate(sorted_indices, 1):
        final_results.append({
            'rank': rank,
            'rrf_score': rrf_scores[doc_idx],
            'document': doc_info[doc_idx]['document'],
            'dense_rank': doc_info[doc_idx]['dense_rank'],
            'bm25_rank': doc_info[doc_idx]['bm25_rank']
        })

    return final_results


# ============================================================================
# RESPONSE GENERATION
# ============================================================================

def build_context(retrieved_results: List[Dict], max_chars: int = 800) -> str:
    """Build context string from retrieved chunks."""
    context_parts = []
    total_chars = 0

    for result in retrieved_results:
        doc = result['document']
        content = doc['content']

        if total_chars + len(content) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 100:
                context_parts.append(content[:remaining] + "...")
            break

        context_parts.append(content)
        total_chars += len(content)

    return "\n\n".join(context_parts)


def generate_answer(query: str, context: str, llm_model) -> str:
    """Generate answer using TinyLlama."""
    prompt = f"""<|system|>
You are a helpful assistant. Answer questions using ONLY the provided context. Be concise and direct.</s>
<|user|>
Context: {context}

Question: {query}</s>
<|assistant|>
Answer: """

    output = llm_model(
        prompt,
        max_new_tokens=100,
        temperature=0.1,
        top_p=0.9,
        top_k=40,
        repetition_penalty=1.1,
        stop=["</s>", "<|user|>", "<|system|>", "\n\n\n"]
    )

    answer = output.strip()
    for artifact in ["</s>", "<|user|>", "<|system|>", "<|assistant|>", "Answer:"]:
        answer = answer.replace(artifact, "").strip()

    return answer


# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================

def run_rag_pipeline(questions: List[Dict], corpus: List[Dict],
                     embedding_model, llm_model, faiss_index, bm25_index,
                     preprocess_func, method: str = "hybrid", rrf_k: int = 60) -> List[Dict]:
    """Run RAG pipeline on all questions."""
    results = []

    for i, q in enumerate(questions):
        query = q['question']

        # Retrieve based on method
        if method == "dense":
            retrieved = retrieve_dense(query, embedding_model, faiss_index, corpus, k=5)
        elif method == "sparse":
            retrieved = retrieve_bm25(query, bm25_index, corpus, preprocess_func, k=5)
        else:  # hybrid
            retrieved = retrieve_hybrid(
                query, embedding_model, faiss_index, bm25_index,
                corpus, preprocess_func, top_k=10, top_n=5, rrf_k=rrf_k
            )

        # Build context and generate answer
        context = build_context(retrieved)
        answer = generate_answer(query, context, llm_model)

        results.append({
            'question_id': q['id'],
            'question': query,
            'ground_truth': q['ground_truth'],
            'generated_answer': answer,
            'retrieved_chunks': retrieved,
            'source_url': q['source_url'],
            'category': q.get('category', 'unknown')
        })

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(questions)} questions")

    return results


def run_ablation_study(questions: List[Dict], corpus: List[Dict],
                       embedding_model, llm_model, faiss_index, bm25_index,
                       preprocess_func) -> Dict[str, Dict]:
    """Run ablation study comparing different retrieval methods."""
    print("\n" + "="*60)
    print("ABLATION STUDY")
    print("="*60)

    ablation_configs = [
        ("Dense-only", "dense", 60),
        ("Sparse-only", "sparse", 60),
        ("Hybrid (k=30)", "hybrid", 30),
        ("Hybrid (k=60)", "hybrid", 60),
        ("Hybrid (k=100)", "hybrid", 100),
    ]

    ablation_results = {}

    for name, method, rrf_k in ablation_configs:
        print(f"\n[{name}] Running evaluation...")

        # Run RAG
        results = run_rag_pipeline(
            questions, corpus, embedding_model, llm_model,
            faiss_index, bm25_index, preprocess_func,
            method=method, rrf_k=rrf_k
        )

        # Calculate metrics
        generated = [r['generated_answer'] for r in results]
        references = [r['ground_truth'] for r in results]

        mrr = calculate_mrr(questions, results)
        rouge = calculate_rouge_l(generated, references)
        bert = calculate_bert_score(generated, references)

        ablation_results[name] = {
            'method': method,
            'rrf_k': rrf_k,
            'mrr': mrr['mrr'],
            'mrr_interpretation': mrr['interpretation'],
            'rouge_l_f1': rouge['mean_f1'],
            'bert_score_f1': bert['mean_f1'],
            'top1_accuracy': mrr['statistics']['top1_accuracy'],
            'top3_accuracy': mrr['statistics']['top3_accuracy']
        }

        print(f"  MRR: {mrr['mrr']:.4f}, ROUGE-L: {rouge['mean_f1']:.4f}, BERTScore: {bert['mean_f1']:.4f}")

    return ablation_results


def perform_error_analysis(results: List[Dict], mrr_details: List[Dict]) -> Dict:
    """Analyze errors and categorize failures."""
    retrieval_failures = []
    generation_failures = []
    category_performance = {}

    for i, result in enumerate(results):
        category = result.get('category', 'unknown')
        if category not in category_performance:
            category_performance[category] = {'total': 0, 'retrieval_success': 0, 'good_answer': 0}

        category_performance[category]['total'] += 1

        # Check retrieval success
        mrr_detail = mrr_details[i] if i < len(mrr_details) else {}
        rank = mrr_detail.get('rank', 0)

        if rank == 0 or rank == "Not found":
            retrieval_failures.append({
                'question_id': result['question_id'],
                'question': result['question'],
                'category': category
            })
        else:
            category_performance[category]['retrieval_success'] += 1

            # Check answer quality (simple heuristic: answer length and overlap)
            gen = result['generated_answer'].lower()
            ref = result['ground_truth'].lower()

            # Simple overlap check
            gen_words = set(gen.split())
            ref_words = set(ref.split())
            overlap = len(gen_words & ref_words) / max(len(ref_words), 1)

            if overlap < 0.2 or len(gen) < 10:
                generation_failures.append({
                    'question_id': result['question_id'],
                    'question': result['question'],
                    'generated': result['generated_answer'][:100],
                    'reference': result['ground_truth'][:100],
                    'category': category
                })
            else:
                category_performance[category]['good_answer'] += 1

    return {
        'retrieval_failures': retrieval_failures,
        'generation_failures': generation_failures,
        'category_performance': category_performance,
        'summary': {
            'total_questions': len(results),
            'retrieval_failure_count': len(retrieval_failures),
            'generation_failure_count': len(generation_failures)
        }
    }


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_html_report(results: List[Dict], metrics: Dict, ablation: Dict,
                         error_analysis: Dict, output_path: str):
    """Generate comprehensive HTML evaluation report."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO

    def fig_to_base64(fig):
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    # Create visualizations
    # 1. Ablation comparison chart
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    methods = list(ablation.keys())
    mrr_scores = [ablation[m]['mrr'] for m in methods]
    rouge_scores = [ablation[m]['rouge_l_f1'] for m in methods]
    bert_scores = [ablation[m]['bert_score_f1'] for m in methods]

    x = np.arange(len(methods))
    width = 0.25

    ax1.bar(x - width, mrr_scores, width, label='MRR', color='#2ecc71')
    ax1.bar(x, rouge_scores, width, label='ROUGE-L', color='#3498db')
    ax1.bar(x + width, bert_scores, width, label='BERTScore', color='#9b59b6')

    ax1.set_ylabel('Score')
    ax1.set_title('Ablation Study: Method Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 1)
    plt.tight_layout()
    ablation_chart = fig_to_base64(fig1)
    plt.close(fig1)

    # 2. Category performance chart
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    categories = list(error_analysis['category_performance'].keys())
    cat_totals = [error_analysis['category_performance'][c]['total'] for c in categories]
    cat_success = [error_analysis['category_performance'][c]['retrieval_success'] for c in categories]

    x = np.arange(len(categories))
    ax2.bar(x, cat_totals, label='Total', color='#bdc3c7', alpha=0.7)
    ax2.bar(x, cat_success, label='Retrieval Success', color='#27ae60')
    ax2.set_ylabel('Count')
    ax2.set_title('Performance by Question Category')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=45, ha='right')
    ax2.legend()
    plt.tight_layout()
    category_chart = fig_to_base64(fig2)
    plt.close(fig2)

    # 3. Score distribution
    fig3, axes = plt.subplots(1, 3, figsize=(12, 4))

    # MRR distribution (reciprocal ranks)
    rr_values = metrics['mrr']['reciprocal_ranks']
    axes[0].hist(rr_values, bins=20, color='#2ecc71', edgecolor='white')
    axes[0].set_title(f"MRR Distribution\n(Mean: {metrics['mrr']['mrr']:.3f})")
    axes[0].set_xlabel('Reciprocal Rank')

    # ROUGE-L distribution
    rouge_values = [s['f1'] for s in metrics['rouge_l']['scores']]
    axes[1].hist(rouge_values, bins=20, color='#3498db', edgecolor='white')
    axes[1].set_title(f"ROUGE-L F1 Distribution\n(Mean: {metrics['rouge_l']['mean_f1']:.3f})")
    axes[1].set_xlabel('F1 Score')

    # BERTScore distribution
    bert_values = [s['f1'] for s in metrics['bert_score']['scores']]
    axes[2].hist(bert_values, bins=20, color='#9b59b6', edgecolor='white')
    axes[2].set_title(f"BERTScore F1 Distribution\n(Mean: {metrics['bert_score']['mean_f1']:.3f})")
    axes[2].set_xlabel('F1 Score')

    plt.tight_layout()
    distribution_chart = fig_to_base64(fig3)
    plt.close(fig3)

    # Build results table (first 20)
    results_rows = ""
    for i, r in enumerate(results[:20]):
        rouge_f1 = metrics['rouge_l']['scores'][i]['f1'] if i < len(metrics['rouge_l']['scores']) else 0
        bert_f1 = metrics['bert_score']['scores'][i]['f1'] if i < len(metrics['bert_score']['scores']) else 0
        rr = metrics['mrr']['reciprocal_ranks'][i] if i < len(metrics['mrr']['reciprocal_ranks']) else 0

        results_rows += f"""
        <tr>
            <td>{r['question_id']}</td>
            <td>{r['category']}</td>
            <td title="{r['question']}">{r['question'][:50]}...</td>
            <td title="{r['ground_truth']}">{r['ground_truth'][:40]}...</td>
            <td title="{r['generated_answer']}">{r['generated_answer'][:40]}...</td>
            <td>{rr:.3f}</td>
            <td>{rouge_f1:.3f}</td>
            <td>{bert_f1:.3f}</td>
        </tr>
        """

    # Build ablation table
    ablation_rows = ""
    for method, data in ablation.items():
        ablation_rows += f"""
        <tr>
            <td>{method}</td>
            <td>{data['mrr']:.4f}</td>
            <td>{data['rouge_l_f1']:.4f}</td>
            <td>{data['bert_score_f1']:.4f}</td>
            <td>{data['top1_accuracy']*100:.1f}%</td>
            <td>{data['top3_accuracy']*100:.1f}%</td>
        </tr>
        """

    # Generate HTML
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Evaluation Report - Group 122</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #2980b9;
            margin-top: 30px;
        }}
        h3 {{
            color: #7f8c8d;
        }}
        .metric-card {{
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 30px;
            border-radius: 10px;
            margin: 10px;
            text-align: center;
            min-width: 150px;
        }}
        .metric-card.green {{ background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }}
        .metric-card.blue {{ background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }}
        .metric-card.purple {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
        }}
        .metric-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #3498db;
            color: white;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .chart {{
            text-align: center;
            margin: 20px 0;
        }}
        .chart img {{
            max-width: 100%;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .justification {{
            background: #e8f4f8;
            padding: 15px;
            border-left: 4px solid #3498db;
            margin: 15px 0;
        }}
        .error-item {{
            background: #ffeaa7;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
        }}
        .footer {{
            text-align: center;
            color: #7f8c8d;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Hybrid RAG Evaluation Report</h1>
        <p><strong>Conversational AI Assignment 2 - Group 122</strong></p>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

        <h2>1. Performance Summary</h2>
        <div style="text-align: center;">
            <div class="metric-card green">
                <div class="metric-value">{metrics['mrr']['mrr']:.3f}</div>
                <div class="metric-label">MRR ({metrics['mrr']['interpretation']})</div>
            </div>
            <div class="metric-card blue">
                <div class="metric-value">{metrics['rouge_l']['mean_f1']:.3f}</div>
                <div class="metric-label">ROUGE-L F1</div>
            </div>
            <div class="metric-card purple">
                <div class="metric-value">{metrics['bert_score']['mean_f1']:.3f}</div>
                <div class="metric-label">BERTScore F1</div>
            </div>
        </div>

        <h2>2. Metric Justifications</h2>

        <h3>2.1 MRR (Mean Reciprocal Rank)</h3>
        <div class="justification">
            <strong>Purpose:</strong> Measures retrieval quality by evaluating how highly the correct source document is ranked.<br>
            <strong>Why MRR:</strong> Standard metric for retrieval systems; penalizes results where correct documents appear lower in rankings.
        </div>

        <h3>2.2 ROUGE-L</h3>
        <div class="justification">
            {metrics['rouge_l']['justification']}
        </div>

        <h3>2.3 BERTScore</h3>
        <div class="justification">
            {metrics['bert_score']['justification']}
        </div>

        <h2>3. Score Distributions</h2>
        <div class="chart">
            <img src="data:image/png;base64,{distribution_chart}" alt="Score Distributions">
        </div>

        <h2>4. Ablation Study</h2>
        <div class="chart">
            <img src="data:image/png;base64,{ablation_chart}" alt="Ablation Study">
        </div>

        <table>
            <tr>
                <th>Method</th>
                <th>MRR</th>
                <th>ROUGE-L</th>
                <th>BERTScore</th>
                <th>Top-1 Acc</th>
                <th>Top-3 Acc</th>
            </tr>
            {ablation_rows}
        </table>

        <h2>5. Performance by Category</h2>
        <div class="chart">
            <img src="data:image/png;base64,{category_chart}" alt="Category Performance">
        </div>

        <h2>6. Detailed Results (Sample)</h2>
        <table>
            <tr>
                <th>ID</th>
                <th>Category</th>
                <th>Question</th>
                <th>Ground Truth</th>
                <th>Generated</th>
                <th>RR</th>
                <th>ROUGE</th>
                <th>BERT</th>
            </tr>
            {results_rows}
        </table>

        <h2>7. Error Analysis</h2>
        <p><strong>Total Questions:</strong> {error_analysis['summary']['total_questions']}</p>
        <p><strong>Retrieval Failures:</strong> {error_analysis['summary']['retrieval_failure_count']} (source document not in top-5)</p>
        <p><strong>Generation Failures:</strong> {error_analysis['summary']['generation_failure_count']} (low answer quality despite correct retrieval)</p>

        <h3>Sample Retrieval Failures</h3>
        {"".join([f'<div class="error-item"><strong>{f["question_id"]}</strong> ({f["category"]}): {f["question"][:80]}...</div>' for f in error_analysis['retrieval_failures'][:5]])}

        <h3>Sample Generation Failures</h3>
        {"".join([f'<div class="error-item"><strong>{f["question_id"]}</strong>: Expected: "{f["reference"]}" | Got: "{f["generated"]}"</div>' for f in error_analysis['generation_failures'][:5]])}

        <div class="footer">
            <p>Conversational AI Assignment 2 - Hybrid RAG Evaluation</p>
            <p>Using TinyLlama-1.1B-Chat with Dense + BM25 + RRF Retrieval</p>
        </div>
    </div>
</body>
</html>
    """

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Report saved to: {output_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*70)
    print("PART 2: AUTOMATED EVALUATION PIPELINE")
    print("Conversational AI Assignment 2 - Group 122")
    print("="*70)

    start_time = time.time()

    # Step 1: Load questions
    print("\n[1/7] Loading evaluation questions...")
    if not os.path.exists(QUESTIONS_FILE):
        print(f"ERROR: {QUESTIONS_FILE} not found!")
        print("Please run Part2_QuestionGen.py first.")
        return

    with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions")

    # Step 2: Load corpus and models
    print("\n[2/7] Loading corpus and models...")
    corpus = load_corpus()
    embedding_model, llm_model = load_models()

    # Step 3: Build indices
    print("\n[3/7] Building retrieval indices...")
    faiss_index, embeddings, bm25_index, preprocess_func = build_indices(corpus, embedding_model)

    # Step 4: Run main evaluation (hybrid with k=60)
    print("\n[4/7] Running RAG evaluation (Hybrid, RRF k=60)...")
    results = run_rag_pipeline(
        questions, corpus, embedding_model, llm_model,
        faiss_index, bm25_index, preprocess_func,
        method="hybrid", rrf_k=60
    )

    # Step 5: Calculate metrics
    print("\n[5/7] Calculating metrics...")
    generated_answers = [r['generated_answer'] for r in results]
    reference_answers = [r['ground_truth'] for r in results]

    print("  Computing MRR...")
    mrr_results = calculate_mrr(questions, results)

    print("  Computing ROUGE-L...")
    rouge_results = calculate_rouge_l(generated_answers, reference_answers)

    print("  Computing BERTScore...")
    bert_results = calculate_bert_score(generated_answers, reference_answers)

    metrics = {
        'mrr': mrr_results,
        'rouge_l': rouge_results,
        'bert_score': bert_results
    }

    print(f"\n  MRR: {mrr_results['mrr']:.4f} ({mrr_results['interpretation']})")
    print(f"  ROUGE-L F1: {rouge_results['mean_f1']:.4f}")
    print(f"  BERTScore F1: {bert_results['mean_f1']:.4f}")

    # Step 6: Run ablation study
    print("\n[6/7] Running ablation study...")
    ablation_results = run_ablation_study(
        questions, corpus, embedding_model, llm_model,
        faiss_index, bm25_index, preprocess_func
    )

    # Step 7: Error analysis and report generation
    print("\n[7/7] Performing error analysis and generating reports...")
    error_analysis = perform_error_analysis(results, mrr_results['details'])

    # Save CSV
    df = pd.DataFrame([{
        'question_id': r['question_id'],
        'category': r['category'],
        'question': r['question'],
        'ground_truth': r['ground_truth'],
        'generated_answer': r['generated_answer'],
        'source_url': r['source_url'],
        'reciprocal_rank': mrr_results['reciprocal_ranks'][i],
        'rouge_l_f1': rouge_results['scores'][i]['f1'],
        'bert_score_f1': bert_results['scores'][i]['f1']
    } for i, r in enumerate(results)])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved: {OUTPUT_CSV}")

    # Save ablation JSON
    with open(OUTPUT_ABLATION, 'w', encoding='utf-8') as f:
        json.dump(ablation_results, f, indent=2)
    print(f"  Saved: {OUTPUT_ABLATION}")

    # Generate HTML report
    generate_html_report(results, metrics, ablation_results, error_analysis, OUTPUT_REPORT)

    # Final summary
    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    print(f"\nResults Summary:")
    print(f"  MRR:         {mrr_results['mrr']:.4f} ({mrr_results['interpretation']})")
    print(f"  ROUGE-L F1:  {rouge_results['mean_f1']:.4f}")
    print(f"  BERTScore F1:{bert_results['mean_f1']:.4f}")
    print(f"\nOutput Files:")
    print(f"  - {OUTPUT_CSV}")
    print(f"  - {OUTPUT_ABLATION}")
    print(f"  - {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
