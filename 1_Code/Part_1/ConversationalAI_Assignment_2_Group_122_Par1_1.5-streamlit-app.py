"""
Conversational AI - RAG System with Streamlit Frontend
=======================================================
A Streamlit web application for the Hybrid RAG (Retrieval-Augmented Generation) system.

This app provides:
- Dense Vector Retrieval (sentence-transformers + FAISS)
- Sparse Keyword Retrieval (BM25)
- Reciprocal Rank Fusion (RRF) for hybrid retrieval
- Response Generation using TinyLlama-1.1B-Chat

Usage:
    streamlit run app.py
"""

import streamlit as st
import json
import os
import time
import string

# Set page config - must be the first Streamlit command
st.set_page_config(
    page_title="Hybrid RAG Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CACHING FUNCTIONS FOR MODEL LOADING
# ============================================================================

@st.cache_resource
def load_sentence_transformer():
    """Load and cache the sentence transformer model."""
    import certifi
    os.environ['SSL_CERT_FILE'] = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    os.environ['CURL_CA_BUNDLE'] = certifi.where()
    
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model


@st.cache_resource
def load_llm_model():
    """Load and cache the TinyLlama model."""
    from ctransformers import AutoModelForCausalLM
    
    LLM_MODEL_REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    LLM_MODEL_FILE = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    MAX_CONTEXT_LENGTH = 2048
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
    return model


@st.cache_data
def load_corpus():
    """Load and cache the document corpus from JSON files."""
    all_documents = []
    
    # Load fixed URL dataset
    if os.path.exists('fixed_url.json'):
        with open('fixed_url.json', 'r', encoding='utf-8') as f:
            fixed_data = json.load(f)
            all_documents.extend(fixed_data)
    
    # Load random URL dataset
    if os.path.exists('random_url.json'):
        with open('random_url.json', 'r', encoding='utf-8') as f:
            random_data = json.load(f)
            all_documents.extend(random_data)
    
    return all_documents


@st.cache_resource
def build_faiss_index(_corpus, _embedding_model):
    """Build and cache the FAISS index."""
    import faiss
    import numpy as np
    
    texts = [doc['content'] for doc in _corpus]
    embeddings = _embedding_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)
    
    return index, embeddings


@st.cache_resource
def build_bm25_index(_corpus):
    """Build and cache the BM25 index."""
    from rank_bm25 import BM25Okapi
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    
    # Download NLTK data silently
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    
    stop_words = set(stopwords.words('english'))
    
    def preprocess(text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
        return tokens
    
    tokenized_corpus = [preprocess(doc['content']) for doc in _corpus]
    bm25_index = BM25Okapi(tokenized_corpus)
    
    return bm25_index, tokenized_corpus, preprocess


# ============================================================================
# RETRIEVAL FUNCTIONS
# ============================================================================

def retrieve_top_k_dense(query, model, index, documents, k=5):
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


def retrieve_top_k_bm25(query, bm25_index, documents, preprocess_func, k=5):
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


def reciprocal_rank_fusion(query, embedding_model, faiss_index, bm25_index, 
                           corpus, preprocess_func, top_k=10, top_n=5, rrf_k=60):
    """Combine dense and sparse retrieval using RRF."""
    
    dense_results = retrieve_top_k_dense(query, embedding_model, faiss_index, corpus, k=top_k)
    bm25_results = retrieve_top_k_bm25(query, bm25_index, corpus, preprocess_func, k=top_k)
    
    rrf_scores = {}
    doc_info = {}
    
    # Process Dense results
    for result in dense_results:
        doc = result['document']
        doc_idx = corpus.index(doc) if doc in corpus else None
        
        if doc_idx is not None:
            rank = result['rank']
            rrf_contribution = 1.0 / (rrf_k + rank)
            
            if doc_idx not in rrf_scores:
                rrf_scores[doc_idx] = 0.0
                doc_info[doc_idx] = {
                    'document': doc,
                    'dense_rank': None,
                    'dense_score': None,
                    'bm25_rank': None,
                    'bm25_score': None
                }
            
            rrf_scores[doc_idx] += rrf_contribution
            doc_info[doc_idx]['dense_rank'] = rank
            doc_info[doc_idx]['dense_score'] = result['similarity_score']
    
    # Process BM25 results
    for result in bm25_results:
        doc = result['document']
        doc_idx = corpus.index(doc) if doc in corpus else None
        
        if doc_idx is not None:
            rank = result['rank']
            rrf_contribution = 1.0 / (rrf_k + rank)
            
            if doc_idx not in rrf_scores:
                rrf_scores[doc_idx] = 0.0
                doc_info[doc_idx] = {
                    'document': doc,
                    'dense_rank': None,
                    'dense_score': None,
                    'bm25_rank': None,
                    'bm25_score': None
                }
            
            rrf_scores[doc_idx] += rrf_contribution
            doc_info[doc_idx]['bm25_rank'] = rank
            doc_info[doc_idx]['bm25_score'] = result['bm25_score']
    
    # Sort and select top-N
    sorted_doc_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_n]
    
    final_results = []
    for rank, doc_idx in enumerate(sorted_doc_indices, 1):
        final_results.append({
            'rank': rank,
            'rrf_score': rrf_scores[doc_idx],
            'document': doc_info[doc_idx]['document'],
            'dense_rank': doc_info[doc_idx]['dense_rank'],
            'dense_score': doc_info[doc_idx]['dense_score'],
            'bm25_rank': doc_info[doc_idx]['bm25_rank'],
            'bm25_score': doc_info[doc_idx]['bm25_score']
        })
    
    return final_results, dense_results, bm25_results


# ============================================================================
# RESPONSE GENERATION
# ============================================================================

def build_context_from_chunks(retrieved_results, max_chars=800):
    """Build context string from retrieved chunks."""
    context_parts = []
    total_chars = 0
    
    for result in retrieved_results:
        doc = result['document']
        content = doc['content']
        
        if total_chars + len(content) > max_chars:
            remaining_chars = max_chars - total_chars
            if remaining_chars > 100:
                context_parts.append(content[:remaining_chars] + "...")
            break
        
        context_parts.append(content)
        total_chars += len(content)
    
    return "\n\n".join(context_parts)


def create_prompt(query, context):
    """Create ChatML formatted prompt for TinyLlama."""
    prompt = f"""<|system|>
You are a helpful assistant. Answer questions using ONLY the provided context. If the context doesn't contain the answer, say "I cannot answer based on the provided context." Be concise and direct.</s>
<|user|>
Context: {context}

Question: {query}</s>
<|assistant|>
Answer: """
    return prompt


def generate_response(query, llm_model, embedding_model, faiss_index, bm25_index, 
                      corpus, preprocess_func, top_k=5, top_n=2, max_tokens=80, temperature=0.1):
    """Generate response using the RAG pipeline."""
    start_time = time.time()
    
    # Step 1: Retrieve chunks using RRF
    retrieved_results, dense_results, bm25_results = reciprocal_rank_fusion(
        query, embedding_model, faiss_index, bm25_index, corpus, preprocess_func,
        top_k=top_k, top_n=top_n, rrf_k=60
    )
    retrieval_time = time.time() - start_time
    
    # Step 2: Build context
    context = build_context_from_chunks(retrieved_results, max_chars=600)
    
    # Step 3: Create prompt
    prompt = create_prompt(query, context)
    
    # Step 4: Generate response
    gen_start = time.time()
    
    # Generate using ctransformers - returns only the new tokens
    full_output = llm_model(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        top_k=40,
        repetition_penalty=1.1,
        stop=["</s>", "<|user|>", "<|system|>", "<|assistant|>", "\n\n\n"]
    )
    gen_time = time.time() - gen_start
    
    # Clean up answer - remove any prompt artifacts
    answer = full_output.strip()
    
    # Remove common artifacts
    for artifact in ["</s>", "<|user|>", "<|system|>", "<|assistant|>", "Answer:"]:
        answer = answer.replace(artifact, "").strip()
    
    # Remove the question if it appears at the start of the answer
    if query.lower() in answer.lower()[:len(query)+20]:
        # Find where the actual answer starts
        lines = answer.split('\n')
        answer = '\n'.join(lines[1:]).strip() if len(lines) > 1 else answer
    
    # Ensure answer ends properly
    if answer and not answer.endswith(('.', '!', '?')):
        last_period = max(answer.rfind('.'), answer.rfind('!'), answer.rfind('?'))
        if last_period > len(answer) * 0.3:
            answer = answer[:last_period + 1]
    
    total_time = time.time() - start_time
    
    return {
        'query': query,
        'retrieved_chunks': retrieved_results,
        'dense_results': dense_results,
        'bm25_results': bm25_results,
        'context': context,
        'answer': answer,
        'timing': {
            'retrieval_sec': round(retrieval_time, 2),
            'generation_sec': round(gen_time, 2),
            'total_sec': round(total_time, 2)
        }
    }


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    # Header
    st.title("🤖 Hybrid RAG Question & Answering Interface")
    st.markdown("""
    **Hybrid Retrieval-Augmented Generation** using Dense Vector + BM25 Sparse Retrieval with Reciprocal Rank Fusion
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    top_k = st.sidebar.slider("Top-K retrieval (per method)", 3, 20, 5)
    top_n = st.sidebar.slider("Top-N chunks for context", 1, 5, 2)
    max_tokens = st.sidebar.slider("Max tokens in response", 50, 200, 80)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Info")
    st.sidebar.markdown("""
    - **Embedding**: all-MiniLM-L6-v2
    - **LLM**: TinyLlama-1.1B-Chat
    - **Quantization**: 4-bit GGUF
    """)
    
    # Load models with progress
    with st.spinner("🔄 Loading models... (first time may take a minute)"):
        try:
            # Load corpus
            corpus = load_corpus()
            if not corpus:
                st.error("❌ No corpus found! Please run the notebook first to create fixed_url.json and random_url.json")
                return
            
            # Load models
            embedding_model = load_sentence_transformer()
            llm_model = load_llm_model()
            
            # Build indices
            faiss_index, _ = build_faiss_index(corpus, embedding_model)
            bm25_index, _, preprocess_func = build_bm25_index(corpus)
            
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            st.info("Make sure all dependencies are installed and the corpus JSON files exist.")
            return
    
    # Display corpus info
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### 📚 Corpus: {len(corpus)} documents")
    
    # Main query input
    st.markdown("---")
    query = st.text_input(
        "🔍 Enter your question:",
        placeholder="e.g., Who was Albert Einstein?",
        key="query_input"
    )
    
    # Generate button
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        generate_btn = st.button("🚀 Generate Answer", type="primary")
    with col2:
        if st.button("🗑️ Clear"):
            st.rerun()
    
    # Process query
    if generate_btn and query:
        with st.spinner("🔄 Processing query..."):
            result = generate_response(
                query, llm_model, embedding_model, faiss_index, bm25_index,
                corpus, preprocess_func, top_k=top_k, top_n=top_n, 
                max_tokens=max_tokens, temperature=temperature
            )
        
        # Display results
        st.markdown("---")
        
        # 1. User Query
        st.markdown("### 📝 USER QUERY")
        st.info(result['query'])
        
        # 2. Generated Answer
        st.markdown("### 🤖 GENERATED ANSWER")
        st.success(result['answer'])
        
        # 3. Retrieved Chunks with Scores
        st.markdown("### 📚 TOP RETRIEVED CHUNKS")
        
        for i, chunk in enumerate(result['retrieved_chunks'], 1):
            doc = chunk['document']
            title = doc['url'].split('/')[-1].replace('_', ' ')
            
            with st.expander(f"[{i}] {title}", expanded=(i == 1)):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**📖 Source:** [{doc['url']}]({doc['url']})")
                    st.markdown(f"**🏷️ Domain:** `{doc['domain']}`")
                    st.markdown("**📄 Content Preview:**")
                    content_preview = doc['content'][:300] + "..." if len(doc['content']) > 300 else doc['content']
                    st.text(content_preview)
                
                with col2:
                    st.markdown("**📊 Scores:**")
                    
                    # Dense score
                    dense_rank = chunk.get('dense_rank', 'N/A')
                    dense_score = chunk.get('dense_score')
                    if dense_score is not None:
                        st.metric("Dense (Semantic)", f"Rank {dense_rank}", f"Score: {dense_score:.4f}")
                    else:
                        st.metric("Dense (Semantic)", "Not in top-K", "")
                    
                    # BM25 score
                    bm25_rank = chunk.get('bm25_rank', 'N/A')
                    bm25_score = chunk.get('bm25_score')
                    if bm25_score is not None:
                        st.metric("Sparse (BM25)", f"Rank {bm25_rank}", f"Score: {bm25_score:.4f}")
                    else:
                        st.metric("Sparse (BM25)", "Not in top-K", "")
                    
                    # RRF score
                    rrf_score = chunk.get('rrf_score', 0)
                    st.metric("RRF (Hybrid)", f"{rrf_score:.6f}", "")
        
        # 4. Response Time
        st.markdown("### ⏱️ RESPONSE TIME BREAKDOWN")
        t = result['timing']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Retrieval Time", f"{t['retrieval_sec']:.2f} sec")
        with col2:
            st.metric("Generation Time", f"{t['generation_sec']:.2f} sec")
        with col3:
            st.metric("Total Time", f"{t['total_sec']:.2f} sec", delta=None)
        
        # Progress bar for timing visualization
        total = t['total_sec']
        if total > 0:
            retrieval_pct = t['retrieval_sec'] / total
            generation_pct = t['generation_sec'] / total
            
            st.markdown("**Time Distribution:**")
            col1, col2 = st.columns(2)
            with col1:
                st.progress(retrieval_pct, text=f"Retrieval: {retrieval_pct*100:.1f}%")
            with col2:
                st.progress(generation_pct, text=f"Generation: {generation_pct*100:.1f}%")
    
    elif generate_btn and not query:
        st.warning("⚠️ Please enter a question first!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <small>
            Conversational AI Assignment 2 - Hybrid RAG System<br>
            Using TinyLlama-1.1B-Chat with Dense + BM25 + RRF Retrieval
        </small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
