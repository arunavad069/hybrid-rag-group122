"""
Conversational AI Assignment 2 - Part 2: Evaluation Metrics
============================================================
Implements evaluation metrics for RAG system assessment.

Metrics:
- MRR (Mean Reciprocal Rank) - Mandatory, measures retrieval quality
- ROUGE-L - Measures longest common subsequence between answers
- BERTScore - Measures semantic similarity using BERT embeddings

Usage:
    from ConversationalAI_Assignment_2_Group_122_Part2_Metrics import (
        calculate_mrr,
        calculate_rouge_l,
        calculate_bert_score,
        calculate_all_metrics
    )
"""

from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# MRR (MEAN RECIPROCAL RANK) - MANDATORY (2 marks)
# ============================================================================

def calculate_mrr(questions: List[Dict], rag_results: List[Dict]) -> Dict[str, Any]:
    """
    Calculate Mean Reciprocal Rank (MRR) at URL level.

    MRR measures how well the retrieval system ranks the correct source document.
    For each question, we find the rank of the correct source URL among retrieved chunks.

    Formula: MRR = (1/|Q|) * sum(1/rank_i) for i in 1..|Q|

    Args:
        questions: List of question dicts with 'source_url' field
        rag_results: List of RAG result dicts with 'retrieved_chunks' containing document URLs

    Returns:
        Dict with 'mrr' (overall score), 'reciprocal_ranks' (per-question scores),
        'ranks' (raw ranks), and 'interpretation' (quality assessment)

    Interpretation:
        MRR > 0.8: Excellent retrieval quality
        MRR 0.6-0.8: Good retrieval quality
        MRR 0.4-0.6: Moderate retrieval quality
        MRR < 0.4: Poor retrieval quality
    """
    reciprocal_ranks = []
    ranks = []
    details = []

    for q, result in zip(questions, rag_results):
        correct_url = q["source_url"]

        # Extract URLs from retrieved chunks
        retrieved_chunks = result.get("retrieved_chunks", [])
        retrieved_urls = []

        for chunk in retrieved_chunks:
            if isinstance(chunk, dict):
                doc = chunk.get("document", {})
                if isinstance(doc, dict):
                    url = doc.get("url", "")
                else:
                    url = ""
            else:
                url = ""
            retrieved_urls.append(url)

        # Find rank of correct URL (1-indexed)
        rank = 0
        for i, url in enumerate(retrieved_urls, 1):
            if url == correct_url:
                rank = i
                break

        # Calculate reciprocal rank
        rr = 1.0 / rank if rank > 0 else 0.0
        reciprocal_ranks.append(rr)
        ranks.append(rank)

        details.append({
            "question_id": q.get("id", "unknown"),
            "correct_url": correct_url.split("/")[-1],
            "rank": rank if rank > 0 else "Not found",
            "reciprocal_rank": rr
        })

    # Calculate MRR
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    # Interpretation
    if mrr > 0.8:
        interpretation = "Excellent"
    elif mrr > 0.6:
        interpretation = "Good"
    elif mrr > 0.4:
        interpretation = "Moderate"
    else:
        interpretation = "Poor"

    # Statistics
    found_count = sum(1 for r in ranks if r > 0)
    top1_count = sum(1 for r in ranks if r == 1)
    top3_count = sum(1 for r in ranks if 0 < r <= 3)
    top5_count = sum(1 for r in ranks if 0 < r <= 5)

    return {
        "mrr": mrr,
        "interpretation": interpretation,
        "reciprocal_ranks": reciprocal_ranks,
        "ranks": ranks,
        "statistics": {
            "total_questions": len(questions),
            "found_in_results": found_count,
            "not_found": len(questions) - found_count,
            "top1_accuracy": top1_count / len(questions) if questions else 0,
            "top3_accuracy": top3_count / len(questions) if questions else 0,
            "top5_accuracy": top5_count / len(questions) if questions else 0
        },
        "details": details
    }


# ============================================================================
# ROUGE-L - CUSTOM METRIC 1 (2 marks)
# ============================================================================

def calculate_rouge_l(generated_answers: List[str], reference_answers: List[str]) -> Dict[str, Any]:
    """
    Calculate ROUGE-L scores for generated answers vs ground truth.

    ROUGE-L measures the longest common subsequence (LCS) between the generated
    and reference answers. It captures:
    - Word overlap: How many words appear in both answers
    - Order preservation: Whether words appear in similar order
    - Answer completeness: How much of the reference is covered

    Justification for RAG evaluation:
    - Captures lexical similarity between generated and expected answers
    - F1 score balances precision (conciseness) and recall (completeness)
    - Uses stemming to handle morphological variations

    Formula: ROUGE-L F1 = 2 * (P_lcs * R_lcs) / (P_lcs + R_lcs)
    where P_lcs = LCS(gen, ref) / len(gen), R_lcs = LCS(gen, ref) / len(ref)

    Args:
        generated_answers: List of generated answer strings
        reference_answers: List of ground truth answer strings

    Returns:
        Dict with 'mean_f1', 'mean_precision', 'mean_recall', 'scores' (per-answer)
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    scores = []
    precisions = []
    recalls = []
    f1_scores = []

    for gen, ref in zip(generated_answers, reference_answers):
        # Handle empty strings
        if not gen or not ref:
            scores.append({
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0
            })
            precisions.append(0.0)
            recalls.append(0.0)
            f1_scores.append(0.0)
            continue

        result = scorer.score(ref, gen)
        rouge_l = result['rougeL']

        scores.append({
            "precision": rouge_l.precision,
            "recall": rouge_l.recall,
            "f1": rouge_l.fmeasure
        })
        precisions.append(rouge_l.precision)
        recalls.append(rouge_l.recall)
        f1_scores.append(rouge_l.fmeasure)

    # Calculate means
    n = len(generated_answers)
    mean_precision = sum(precisions) / n if n > 0 else 0.0
    mean_recall = sum(recalls) / n if n > 0 else 0.0
    mean_f1 = sum(f1_scores) / n if n > 0 else 0.0

    # Interpretation
    if mean_f1 > 0.5:
        interpretation = "High lexical overlap"
    elif mean_f1 > 0.3:
        interpretation = "Moderate lexical overlap"
    else:
        interpretation = "Low lexical overlap"

    return {
        "mean_f1": mean_f1,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "interpretation": interpretation,
        "scores": scores,
        "justification": (
            "ROUGE-L measures longest common subsequence between generated and reference answers. "
            "It captures word overlap, order preservation, and answer completeness. "
            "The F1 score balances precision (conciseness) and recall (coverage)."
        )
    }


# ============================================================================
# BERTSCORE - CUSTOM METRIC 2 (2 marks)
# ============================================================================

def calculate_bert_score(generated_answers: List[str], reference_answers: List[str],
                         model_type: str = "distilbert-base-uncased") -> Dict[str, Any]:
    """
    Calculate BERTScore for semantic similarity between answers.

    BERTScore uses contextual embeddings from BERT to measure semantic similarity,
    capturing meaning beyond surface-level word matching. It handles:
    - Synonyms: "car" and "automobile" are recognized as similar
    - Paraphrases: Different phrasings of the same meaning
    - Semantic equivalence: Answers that mean the same but differ lexically

    Justification for RAG evaluation:
    - Correlates highly with human judgment of answer quality
    - Captures semantic equivalence even with different wording
    - More robust than n-gram based metrics for evaluating free-form answers

    Args:
        generated_answers: List of generated answer strings
        reference_answers: List of ground truth answer strings
        model_type: BERT model to use (default: distilbert-base-uncased for speed)

    Returns:
        Dict with 'mean_f1', 'mean_precision', 'mean_recall', 'scores' (per-answer)
    """
    from bert_score import score as bert_score_func

    # Handle empty inputs
    if not generated_answers or not reference_answers:
        return {
            "mean_f1": 0.0,
            "mean_precision": 0.0,
            "mean_recall": 0.0,
            "interpretation": "No answers to evaluate",
            "scores": [],
            "justification": ""
        }

    # Replace empty strings with placeholder to avoid errors
    gen_clean = [g if g.strip() else "no answer" for g in generated_answers]
    ref_clean = [r if r.strip() else "no reference" for r in reference_answers]

    # Calculate BERTScore
    P, R, F1 = bert_score_func(
        gen_clean,
        ref_clean,
        model_type=model_type,
        lang="en",
        verbose=False,
        rescale_with_baseline=True  # Normalize scores to be more interpretable
    )

    # Convert tensors to lists
    precisions = P.tolist()
    recalls = R.tolist()
    f1_scores = F1.tolist()

    # Build per-answer scores
    scores = []
    for p, r, f in zip(precisions, recalls, f1_scores):
        scores.append({
            "precision": p,
            "recall": r,
            "f1": f
        })

    # Calculate means
    mean_precision = sum(precisions) / len(precisions)
    mean_recall = sum(recalls) / len(recalls)
    mean_f1 = sum(f1_scores) / len(f1_scores)

    # Interpretation
    if mean_f1 > 0.7:
        interpretation = "High semantic similarity"
    elif mean_f1 > 0.5:
        interpretation = "Moderate semantic similarity"
    else:
        interpretation = "Low semantic similarity"

    return {
        "mean_f1": mean_f1,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "interpretation": interpretation,
        "model_used": model_type,
        "scores": scores,
        "justification": (
            "BERTScore uses contextual BERT embeddings to measure semantic similarity. "
            "It captures synonyms, paraphrases, and semantic equivalence beyond lexical matching. "
            "Research shows it correlates highly with human judgment of text quality."
        )
    }


# ============================================================================
# COMBINED METRICS CALCULATION
# ============================================================================

def calculate_all_metrics(
    questions: List[Dict],
    rag_results: List[Dict],
    generated_answers: List[str],
    reference_answers: List[str]
) -> Dict[str, Any]:
    """
    Calculate all evaluation metrics in one call.

    Args:
        questions: List of question dicts with source URLs
        rag_results: List of RAG results with retrieved chunks
        generated_answers: List of generated answer strings
        reference_answers: List of ground truth answer strings

    Returns:
        Dict containing results for all three metrics
    """
    print("Calculating MRR...")
    mrr_results = calculate_mrr(questions, rag_results)

    print("Calculating ROUGE-L...")
    rouge_results = calculate_rouge_l(generated_answers, reference_answers)

    print("Calculating BERTScore...")
    bert_results = calculate_bert_score(generated_answers, reference_answers)

    return {
        "mrr": mrr_results,
        "rouge_l": rouge_results,
        "bert_score": bert_results,
        "summary": {
            "MRR": f"{mrr_results['mrr']:.4f} ({mrr_results['interpretation']})",
            "ROUGE-L F1": f"{rouge_results['mean_f1']:.4f} ({rouge_results['interpretation']})",
            "BERTScore F1": f"{bert_results['mean_f1']:.4f} ({bert_results['interpretation']})"
        }
    }


# ============================================================================
# LLM-AS-JUDGE (BONUS)
# ============================================================================

def llm_judge_score(model, question: str, generated: str, reference: str) -> Dict[str, int]:
    """
    Use TinyLlama as a judge to rate answer quality on 4 dimensions.

    Dimensions (1-5 scale):
    - Accuracy: Is the information correct?
    - Completeness: Does it fully answer the question?
    - Relevance: Is the answer on-topic?
    - Coherence: Is it well-structured and clear?

    Args:
        model: TinyLlama model instance
        question: The original question
        generated: The generated answer
        reference: The ground truth answer

    Returns:
        Dict with scores for each dimension
    """
    prompt = f"""<|system|>
You are an answer quality evaluator. Rate the generated answer on a scale of 1-5 for each criterion.
1 = Very Poor, 2 = Poor, 3 = Average, 4 = Good, 5 = Excellent</s>
<|user|>
Question: {question}
Reference Answer: {reference}
Generated Answer: {generated}

Rate the generated answer:
- Accuracy (1-5): Is information correct?
- Completeness (1-5): Fully answers question?
- Relevance (1-5): On-topic?
- Coherence (1-5): Well-structured?

Output format: Accuracy:X Completeness:X Relevance:X Coherence:X</s>
<|assistant|>
"""

    try:
        output = model(
            prompt,
            max_new_tokens=50,
            temperature=0.1,
            stop=["</s>", "<|user|>", "\n\n"]
        )

        # Parse scores from output
        scores = {"accuracy": 3, "completeness": 3, "relevance": 3, "coherence": 3}
        output = output.lower()

        for dim in scores.keys():
            if dim in output:
                # Find the number after the dimension name
                import re
                match = re.search(f"{dim}[:\\s]*([1-5])", output)
                if match:
                    scores[dim] = int(match.group(1))

        return scores

    except Exception as e:
        # Return neutral scores on error
        return {"accuracy": 3, "completeness": 3, "relevance": 3, "coherence": 3}


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("PART 2.2: EVALUATION METRICS - TEST")
    print("="*60)

    # Test data
    test_questions = [
        {"id": "q001", "source_url": "https://en.wikipedia.org/wiki/Python_(programming_language)"},
        {"id": "q002", "source_url": "https://en.wikipedia.org/wiki/Albert_Einstein"},
    ]

    test_results = [
        {
            "retrieved_chunks": [
                {"document": {"url": "https://en.wikipedia.org/wiki/Java_(programming_language)"}},
                {"document": {"url": "https://en.wikipedia.org/wiki/Python_(programming_language)"}},
                {"document": {"url": "https://en.wikipedia.org/wiki/JavaScript"}},
            ]
        },
        {
            "retrieved_chunks": [
                {"document": {"url": "https://en.wikipedia.org/wiki/Albert_Einstein"}},
                {"document": {"url": "https://en.wikipedia.org/wiki/Isaac_Newton"}},
            ]
        }
    ]

    generated = [
        "Python is a high-level programming language known for its readability.",
        "Albert Einstein was a physicist who developed the theory of relativity."
    ]

    references = [
        "Python is an interpreted, high-level programming language with dynamic typing.",
        "Einstein was a German-born theoretical physicist, known for the theory of relativity."
    ]

    # Test MRR
    print("\n[Testing MRR]")
    mrr_result = calculate_mrr(test_questions, test_results)
    print(f"MRR: {mrr_result['mrr']:.4f}")
    print(f"Interpretation: {mrr_result['interpretation']}")
    print(f"Ranks: {mrr_result['ranks']}")

    # Test ROUGE-L
    print("\n[Testing ROUGE-L]")
    rouge_result = calculate_rouge_l(generated, references)
    print(f"Mean F1: {rouge_result['mean_f1']:.4f}")
    print(f"Interpretation: {rouge_result['interpretation']}")

    # Test BERTScore
    print("\n[Testing BERTScore]")
    bert_result = calculate_bert_score(generated, references)
    print(f"Mean F1: {bert_result['mean_f1']:.4f}")
    print(f"Interpretation: {bert_result['interpretation']}")

    print("\n" + "="*60)
    print("All metrics tests completed successfully!")
    print("="*60)
