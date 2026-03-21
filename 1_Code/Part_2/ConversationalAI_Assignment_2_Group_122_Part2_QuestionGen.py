"""
Conversational AI Assignment 2 - Part 2: Question Generation
=============================================================
Generates 100 diverse Q&A pairs from the Wikipedia corpus for RAG evaluation.

Distribution:
- 40 Factual questions (direct facts from documents)
- 20 Comparative questions (comparing entities/concepts)
- 25 Inferential questions (requiring reasoning)
- 15 Multi-hop questions (requiring multiple document pieces)

Output: evaluation_questions.json
"""

import json
import os
import random
import uuid
from typing import List, Dict, Any

# ============================================================================
# CONFIGURATION
# ============================================================================

# Question distribution
QUESTION_DISTRIBUTION = {
    "factual": 40,
    "comparative": 20,
    "inferential": 25,
    "multi_hop": 15
}

# LLM Configuration
LLM_MODEL_REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
LLM_MODEL_FILE = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MAX_CONTEXT_LENGTH = 2048

# File paths
FIXED_URL_FILE = "fixed_url.json"
RANDOM_URL_FILE = "random_url.json"
OUTPUT_FILE = "evaluation_questions.json"


# ============================================================================
# PROMPT TEMPLATES FOR DIFFERENT QUESTION TYPES
# ============================================================================

QUESTION_PROMPTS = {
    "factual": """<|system|>
You are a question generator. Create a simple factual question that can be answered directly from the given text. The question should ask about a specific fact, date, name, or definition.
</s>
<|user|>
Text: {content}

Generate ONE factual question and its answer based on this text. Format:
Q: [question]
A: [answer from text]</s>
<|assistant|>
""",

    "comparative": """<|system|>
You are a question generator. Create a comparative question that asks about differences, similarities, or relationships mentioned in the text.
</s>
<|user|>
Text: {content}

Generate ONE comparative question and its answer based on this text. The question should compare aspects, features, or characteristics. Format:
Q: [question]
A: [answer from text]</s>
<|assistant|>
""",

    "inferential": """<|system|>
You are a question generator. Create an inferential question that requires understanding and reasoning about the text, not just finding a direct fact.
</s>
<|user|>
Text: {content}

Generate ONE inferential question and its answer. The question should require understanding why or how something works based on the text. Format:
Q: [question]
A: [answer from text]</s>
<|assistant|>
""",

    "multi_hop": """<|system|>
You are a question generator. Create a question that connects multiple pieces of information from the text to form an answer.
</s>
<|user|>
Text: {content}

Generate ONE multi-hop question and its answer. The question should require combining multiple facts from the text. Format:
Q: [question]
A: [answer from text]</s>
<|assistant|>
"""
}


# ============================================================================
# FALLBACK QUESTION TEMPLATES (if LLM fails)
# ============================================================================

FALLBACK_TEMPLATES = {
    "factual": [
        "What is {entity} known for?",
        "When was {entity} established/born?",
        "Where is {entity} located?",
        "Who is {entity}?",
        "What does {entity} refer to?"
    ],
    "comparative": [
        "How does {entity} compare to others in its field?",
        "What distinguishes {entity} from similar entities?",
        "What are the main characteristics of {entity}?"
    ],
    "inferential": [
        "Why is {entity} significant?",
        "What can be inferred about {entity} from its history?",
        "How has {entity} influenced its domain?"
    ],
    "multi_hop": [
        "What is {entity} and how is it connected to its field?",
        "Considering the history and characteristics of {entity}, what makes it notable?"
    ]
}


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_llm_model():
    """Load TinyLlama model for question generation."""
    from ctransformers import AutoModelForCausalLM

    print("Loading TinyLlama model...")
    CPU_THREADS = os.cpu_count() or 4

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_REPO,
        model_file=LLM_MODEL_FILE,
        model_type="llama",
        context_length=MAX_CONTEXT_LENGTH,
        gpu_layers=0,
        threads=CPU_THREADS,
        batch_size=512
    )
    print("Model loaded successfully!")
    return model


# ============================================================================
# CORPUS LOADING
# ============================================================================

def load_corpus() -> List[Dict[str, Any]]:
    """Load document corpus from JSON files."""
    all_documents = []

    # Load fixed URL dataset
    if os.path.exists(FIXED_URL_FILE):
        with open(FIXED_URL_FILE, 'r', encoding='utf-8') as f:
            fixed_data = json.load(f)
            all_documents.extend(fixed_data)
            print(f"Loaded {len(fixed_data)} documents from {FIXED_URL_FILE}")
    else:
        print(f"Warning: {FIXED_URL_FILE} not found")

    # Load random URL dataset
    if os.path.exists(RANDOM_URL_FILE):
        with open(RANDOM_URL_FILE, 'r', encoding='utf-8') as f:
            random_data = json.load(f)
            all_documents.extend(random_data)
            print(f"Loaded {len(random_data)} documents from {RANDOM_URL_FILE}")
    else:
        print(f"Warning: {RANDOM_URL_FILE} not found")

    print(f"Total corpus size: {len(all_documents)} documents")
    return all_documents


# ============================================================================
# QUESTION GENERATION
# ============================================================================

def extract_entity_from_url(url: str) -> str:
    """Extract entity name from Wikipedia URL."""
    if '/wiki/' in url:
        entity = url.split('/wiki/')[-1]
        entity = entity.replace('_', ' ')
        # Remove URL encoding
        from urllib.parse import unquote
        entity = unquote(entity)
        return entity
    return "this topic"


def generate_question_with_llm(model, content: str, category: str) -> tuple:
    """Generate a Q&A pair using TinyLlama."""
    prompt = QUESTION_PROMPTS[category].format(content=content[:500])

    try:
        output = model(
            prompt,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            stop=["</s>", "<|user|>", "<|system|>", "\n\n\n"]
        )

        # Parse Q and A from output
        output = output.strip()

        if "Q:" in output and "A:" in output:
            parts = output.split("A:")
            question = parts[0].replace("Q:", "").strip()
            answer = parts[1].strip() if len(parts) > 1 else ""

            # Clean up
            question = question.strip().rstrip("?") + "?"
            answer = answer.split("\n")[0].strip()  # Take first line only

            if len(question) > 10 and len(answer) > 5:
                return question, answer
    except Exception as e:
        print(f"LLM generation error: {e}")

    return None, None


def generate_fallback_question(doc: Dict, category: str) -> tuple:
    """Generate a fallback question when LLM fails."""
    entity = extract_entity_from_url(doc['url'])
    content = doc['content']

    template = random.choice(FALLBACK_TEMPLATES[category])
    question = template.format(entity=entity)

    # Create answer from first sentence of content
    sentences = content.split('.')
    answer = sentences[0].strip() + "." if sentences else content[:100]

    return question, answer


def generate_questions(model, corpus: List[Dict], distribution: Dict[str, int]) -> List[Dict]:
    """Generate Q&A pairs according to the specified distribution."""
    questions = []
    question_id = 1

    # Group documents by domain for diversity
    docs_by_domain = {}
    for doc in corpus:
        domain = doc.get('domain', 'general')
        if domain not in docs_by_domain:
            docs_by_domain[domain] = []
        docs_by_domain[domain].append(doc)

    print(f"\nDomains found: {list(docs_by_domain.keys())}")

    for category, count in distribution.items():
        print(f"\n{'='*50}")
        print(f"Generating {count} {category} questions...")
        print(f"{'='*50}")

        generated = 0
        attempts = 0
        max_attempts = count * 3  # Allow retries

        # Shuffle documents for each category
        available_docs = corpus.copy()
        random.shuffle(available_docs)

        while generated < count and attempts < max_attempts and available_docs:
            doc = available_docs.pop(0)
            attempts += 1

            # Skip documents with too little content
            if len(doc.get('content', '')) < 50:
                continue

            # Try LLM generation first
            question, answer = generate_question_with_llm(model, doc['content'], category)

            # Fallback if LLM fails
            if question is None:
                question, answer = generate_fallback_question(doc, category)

            # Validate and add
            if question and answer and len(question) > 10 and len(answer) > 5:
                qa_entry = {
                    "id": f"q{question_id:03d}",
                    "question": question,
                    "ground_truth": answer,
                    "source_url": doc['url'],
                    "source_id": doc.get('id', str(uuid.uuid4())),
                    "category": category,
                    "domain": doc.get('domain', 'general')
                }
                questions.append(qa_entry)
                generated += 1
                question_id += 1

                print(f"[{generated}/{count}] {category}: {question[:60]}...")

        print(f"Generated {generated}/{count} {category} questions")

    return questions


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*60)
    print("PART 2.1: QUESTION GENERATION")
    print("Conversational AI Assignment 2 - Group 122")
    print("="*60)

    # Step 1: Load corpus
    print("\n[1/3] Loading corpus...")
    corpus = load_corpus()

    if not corpus:
        print("ERROR: No corpus found! Please run Part 1 notebook first.")
        return

    # Step 2: Load LLM
    print("\n[2/3] Loading TinyLlama model...")
    model = load_llm_model()

    # Step 3: Generate questions
    print("\n[3/3] Generating questions...")
    questions = generate_questions(model, corpus, QUESTION_DISTRIBUTION)

    # Summary
    print("\n" + "="*60)
    print("GENERATION SUMMARY")
    print("="*60)

    category_counts = {}
    for q in questions:
        cat = q['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1

    print(f"Total questions generated: {len(questions)}")
    for cat, count in category_counts.items():
        print(f"  - {cat}: {count}")

    # Save to file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to: {OUTPUT_FILE}")

    # Show sample questions
    print("\n" + "="*60)
    print("SAMPLE QUESTIONS")
    print("="*60)
    for i, q in enumerate(random.sample(questions, min(5, len(questions)))):
        print(f"\n[{q['id']}] ({q['category']})")
        print(f"Q: {q['question']}")
        print(f"A: {q['ground_truth'][:100]}...")
        print(f"Source: {q['source_url'].split('/')[-1].replace('_', ' ')}")


if __name__ == "__main__":
    main()
