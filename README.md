# Engineering Oracle — RAG Pipeline Laboratory

A **comprehensive, modular RAG pipeline template** for technical documentation and engineering QA. Every stage offers **multiple strategies** — from naive to state-of-the-art — enabling systematic experimentation and benchmarking.

## Architecture

```
Document → Ingestion → Chunking → Embedding → Vector Store → Retrieval → Generation → Evaluation
  (10)        (9)         (10)        (3)          (12)          (5+3)        (RAGAS)
```

## Pipeline Coverage

| Stage | # Strategies | Range |
|---|---|---|
| **Ingestion** | 11 | TextLoader → LlamaParse (vision-AI) |
| **Chunking** | 9 | Fixed character → Semantic / Parent-child |
| **Embedding** | 10 | OpenAI → BGE / E5 / GTE / Instructor / Nomic / Ollama |
| **Vector Store** | 3 | FAISS, ChromaDB, Qdrant |
| **Retrieval** | 12 | Naive similarity → Cross-encoder rerank / HyDE / Self-query |
| **Generation** | 5 | Basic LCEL → Map-Reduce / Refine / Conversational |
| **Advanced RAG** | 3 | Corrective RAG, Adaptive RAG, Step-Back Prompting |
| **Evaluation** | RAGAS + Custom | Faithfulness, relevancy, latency, utilization |

## Quick Start

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys (only needed for API mode)

# 3. For local mode, ensure Ollama is running
ollama pull llama3
ollama pull nomic-embed-text

# 4. Open in VS Code and run as interactive Python
# The file uses # %% cell markers — VS Code treats it as a Jupyter notebook
```

## Environment Modes

| Mode | Embeddings | LLM | API Keys Needed |
|---|---|---|---|
| `local` | HuggingFace (BGE) | Ollama (Llama 3) | None |
| `api` | OpenAI | OpenAI (GPT-4) | `OPENAI_API_KEY` |
| `hybrid` | HuggingFace (BGE) | OpenAI (GPT-4) | `OPENAI_API_KEY` |

## Usage

### Minimal Pipeline (3 lines)
```python
config = PipelineConfig(ingestion_strategy="pypdf", retrieval_strategy="hybrid")
result = run_pipeline("./data/manual.pdf", query="What is X?", config=config)
print(result["answer"])
```

### Strategy Comparison Experiment
```python
configs = [
    PipelineConfig(retrieval_strategy="naive_similarity"),
    PipelineConfig(retrieval_strategy="hybrid"),
    PipelineConfig(retrieval_strategy="cross_encoder_rerank"),
]
run_experiment("./data/manual.pdf", test_queries=["..."], experiment_configs=configs)
```

## File Structure

```
engineering-oracle/
├── engineering_oracle_rag_pipeline.py  # Main notebook (# %% cell markers)
├── requirements.txt                    # All dependencies
├── .env.example                        # Environment variable template
├── README.md                           # This file
└── data/                               # Place your documents here
```

## License

MIT — Built for the Engineering Oracle Project.
