# %% [markdown]
# # 🔮 Engineering Oracle — Complete RAG Pipeline Laboratory
#
# **A Modular, Strategy-Switchable RAG Pipeline for Technical Documentation & Engineering QA**
#
# This notebook provides a comprehensive, plug-and-play environment for building,
# testing, and benchmarking Retrieval-Augmented Generation (RAG) pipelines.
# Every stage offers **multiple strategies** — from the most naive to the most
# advanced — enabling systematic experimentation and direct comparison.
#
# ---
#
# ## Pipeline Architecture
# ```
# ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌─────────────┐   ┌───────────┐   ┌────────────┐   ┌────────────┐
# │ Document │──▶│ Ingest   │──▶│ Chunk     │──▶│ Embed       │──▶│ Store     │──▶│ Retrieve   │──▶│ Generate   │
# │ Source   │   │ Pipeline │   │ Strategies│   │ Models      │   │ Vectors   │   │ Strategies │   │ + Evaluate │
# └──────────┘   └──────────┘   └───────────┘   └─────────────┘   └───────────┘   └────────────┘   └────────────┘
# ```
#
# ## Table of Contents
# 1. **Environment Setup & Configuration** — Toggle local/API, hardware, model registry
# 2. **Document Ingestion Pipeline** — 10 strategies (text → vision-AI parsing)
# 3. **Chunking Strategies** — 9 strategies (fixed-char → semantic/proposition)
# 4. **Embedding Models** — 8 model options (local HF → API-based)
# 5. **Vector Store Construction** — 3 backends (FAISS, Chroma, Qdrant)
# 6. **Retrieval Strategies** — 12 strategies (naive → ColBERT-style)
# 7. **Generation & Chain Assembly** — 5 chain types (basic → conversational)
# 8. **Advanced RAG Patterns** — HyDE, FLARE, Corrective RAG
# 9. **Evaluation Framework** — RAGAS + custom metrics
# 10. **Use Case Demonstrations** — Engineering-specific scenarios
# 11. **Pipeline Orchestrator** — End-to-end runner with experiment tracking

# %%
# ============================================================
# CELL 1: GLOBAL SETUP & CONFIGURATION
# ============================================================
# This cell manages ALL configuration in one place.
# Change settings here to control the entire pipeline.
# ============================================================

import os
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Literal
from enum import Enum

from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

# ----------------------------------------------------------
# 1A. Configuration Dataclass
# ----------------------------------------------------------

class IngestionStrategy(Enum):
    """All supported document ingestion strategies, ordered naive → advanced."""
    TEXT_LOADER       = "text_loader"        # Plain .txt files
    PYPDF             = "pypdf"              # Basic PDF text extraction
    PYMUPDF           = "pymupdf"            # Better fidelity PDF extraction
    PDFPLUMBER        = "pdfplumber"          # Table-aware PDF extraction
    UNSTRUCTURED      = "unstructured"        # Layout-aware, mixed content
    UNSTRUCTURED_HIRES = "unstructured_hires" # High-resolution strategy (OCR + layout)
    DOCX              = "docx"               # Word documents
    CSV               = "csv"                # Structured data
    WEB               = "web"                # Web pages / online docs
    DIRECTORY         = "directory"           # Batch directory processing
    LLAMA_PARSE       = "llama_parse"         # Vision/AI-based (most advanced)

class ChunkingStrategy(Enum):
    """All supported chunking strategies, ordered naive → advanced."""
    FIXED_CHARACTER   = "fixed_character"     # Hard character split
    RECURSIVE         = "recursive"           # Smart recursive split
    TOKEN_BASED       = "token_based"         # Token-count aware
    SENTENCE          = "sentence"            # Sentence boundary aware (NLTK)
    SPACY_SENTENCE    = "spacy_sentence"      # Sentence boundary aware (spaCy)
    MARKDOWN_HEADER   = "markdown_header"     # Structure-preserving (markdown)
    HTML_HEADER       = "html_header"         # Structure-preserving (HTML)
    SEMANTIC          = "semantic"            # Embedding-based grouping
    PARENT_DOCUMENT   = "parent_document"     # Hierarchical parent/child

class EmbeddingModel(Enum):
    """All supported embedding models."""
    OPENAI_SMALL      = "openai_small"        # text-embedding-3-small  (API)
    OPENAI_LARGE      = "openai_large"        # text-embedding-3-large  (API)
    BGE_LARGE         = "bge_large"           # BAAI/bge-large-en-v1.5  (Local)
    BGE_BASE          = "bge_base"            # BAAI/bge-base-en-v1.5   (Local)
    E5_LARGE          = "e5_large"            # intfloat/e5-large-v2    (Local)
    GTE_LARGE         = "gte_large"           # thenlper/gte-large      (Local)
    INSTRUCTOR_XL     = "instructor_xl"       # hkunlp/instructor-xl    (Local)
    COHERE            = "cohere"              # embed-english-v3.0      (API)
    NOMIC             = "nomic"               # nomic-embed-text-v1.5   (Local)
    OLLAMA            = "ollama"              # Any Ollama model        (Local)

class VectorStoreBackend(Enum):
    """Supported vector store backends."""
    FAISS             = "faiss"
    CHROMA            = "chroma"
    QDRANT            = "qdrant"

class RetrievalStrategy(Enum):
    """All supported retrieval strategies, ordered naive → advanced."""
    NAIVE_SIMILARITY  = "naive_similarity"    # Basic cosine top-k
    MMR               = "mmr"                 # Maximal Marginal Relevance
    BM25              = "bm25"                # Sparse keyword retrieval
    HYBRID            = "hybrid"              # BM25 + Dense ensemble
    MULTI_QUERY       = "multi_query"         # LLM-generated query variants
    HYDE              = "hyde"                 # Hypothetical Document Embeddings
    SELF_QUERY        = "self_query"           # Metadata-aware structured query
    PARENT_DOCUMENT   = "parent_document"     # Retrieve parent context
    CONTEXTUAL_COMPRESSION = "contextual_compression"  # LLM-compressed context
    CROSS_ENCODER_RERANK   = "cross_encoder_rerank"    # Neural re-scoring
    COHERE_RERANK     = "cohere_rerank"       # Cohere API re-ranking
    LONG_CONTEXT_REORDER   = "long_context_reorder"    # Lost-in-the-middle fix

class GenerationMode(Enum):
    """Generation chain types."""
    BASIC             = "basic"               # Simple RAG chain
    STUFF             = "stuff"               # Stuff all docs into prompt
    MAP_REDUCE        = "map_reduce"          # Process docs in parallel, merge
    REFINE            = "refine"              # Iteratively refine with each doc
    CONVERSATIONAL    = "conversational"      # Multi-turn with memory


@dataclass
class PipelineConfig:
    """
    Central configuration for the entire RAG pipeline.
    Modify this single object to reconfigure any stage.
    """
    # --- Environment ---
    environment: Literal["local", "api", "hybrid"] = "local"
    device: str = "cpu"  # "cpu" | "cuda" | "mps"

    # --- Pipeline Stage Selections ---
    ingestion_strategy: str = IngestionStrategy.PYPDF.value
    chunking_strategy: str  = ChunkingStrategy.RECURSIVE.value
    embedding_model: str    = EmbeddingModel.BGE_LARGE.value
    vector_store: str       = VectorStoreBackend.FAISS.value
    retrieval_strategy: str = RetrievalStrategy.HYBRID.value
    generation_mode: str    = GenerationMode.BASIC.value

    # --- Chunking Parameters ---
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # --- Retrieval Parameters ---
    top_k: int = 5               # Final number of documents to retrieve
    fetch_k: int = 20            # Candidate pool size (for MMR, reranking)
    hybrid_weights: List[float] = field(default_factory=lambda: [0.5, 0.5])

    # --- LLM Parameters ---
    llm_model_name: str = "llama3"       # Ollama model for local
    api_model_name: str = "gpt-4-turbo"  # OpenAI model for API
    temperature: float = 0.0

    # --- Paths ---
    persist_directory: str = "./vector_store"
    data_directory: str = "./data"

    def __post_init__(self):
        """Load overrides from environment variables if present."""
        self.environment = os.getenv("RAG_ENVIRONMENT", self.environment)
        self.device = os.getenv("DEVICE", self.device)


# --- Instantiate Global Config ---
config = PipelineConfig()

print("=" * 60)
print("🔮 ENGINEERING ORACLE — Pipeline Configuration")
print("=" * 60)
print(f"  Environment     : {config.environment.upper()}")
print(f"  Device          : {config.device.upper()}")
print(f"  Ingestion       : {config.ingestion_strategy}")
print(f"  Chunking        : {config.chunking_strategy}")
print(f"  Embedding       : {config.embedding_model}")
print(f"  Vector Store    : {config.vector_store}")
print(f"  Retrieval       : {config.retrieval_strategy}")
print(f"  Generation      : {config.generation_mode}")
print("=" * 60)


# %%
# ============================================================
# CELL 2: MODEL INITIALIZATION
# ============================================================
# Initializes embedding and LLM models based on config.
# ============================================================

def initialize_embedding_model(config: PipelineConfig):
    """
    Initialize the embedding model based on the selected strategy.

    Returns:
        An embedding model instance compatible with LangChain.
    """
    model_id = config.embedding_model
    device = config.device
    print(f"⚙️  Initializing embedding model: {model_id} (device={device})")

    # --- OpenAI Embeddings (API) ---
    if model_id in (EmbeddingModel.OPENAI_SMALL.value, EmbeddingModel.OPENAI_LARGE.value):
        from langchain_openai import OpenAIEmbeddings
        model_name = "text-embedding-3-small" if model_id == EmbeddingModel.OPENAI_SMALL.value else "text-embedding-3-large"
        return OpenAIEmbeddings(model=model_name)

    # --- BGE Embeddings (Local) ---
    elif model_id in (EmbeddingModel.BGE_LARGE.value, EmbeddingModel.BGE_BASE.value):
        from langchain_huggingface import HuggingFaceEmbeddings
        hf_model = "BAAI/bge-large-en-v1.5" if model_id == EmbeddingModel.BGE_LARGE.value else "BAAI/bge-base-en-v1.5"
        return HuggingFaceEmbeddings(
            model_name=hf_model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

    # --- E5 Embeddings (Local) ---
    elif model_id == EmbeddingModel.E5_LARGE.value:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="intfloat/e5-large-v2",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

    # --- GTE Embeddings (Local) ---
    elif model_id == EmbeddingModel.GTE_LARGE.value:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="thenlper/gte-large",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

    # --- Instructor Embeddings (Local) ---
    elif model_id == EmbeddingModel.INSTRUCTOR_XL.value:
        from langchain_community.embeddings import HuggingFaceInstructEmbeddings
        return HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-xl",
            model_kwargs={"device": device},
            # Instruction prefix tailored for engineering retrieval
            embed_instruction="Represent the engineering document for retrieval: ",
            query_instruction="Represent the engineering question for retrieving relevant documents: ",
        )

    # --- Cohere Embeddings (API) ---
    elif model_id == EmbeddingModel.COHERE.value:
        from langchain_cohere import CohereEmbeddings
        return CohereEmbeddings(
            model="embed-english-v3.0",
            cohere_api_key=os.getenv("COHERE_API_KEY"),
        )

    # --- Nomic Embeddings (Local / API) ---
    elif model_id == EmbeddingModel.NOMIC.value:
        from langchain_nomic import NomicEmbeddings
        return NomicEmbeddings(
            model="nomic-embed-text-v1.5",
            inference_mode="local",  # Set to "remote" if using API
            device=device,
        )

    # --- Ollama Embeddings (Local) ---
    elif model_id == EmbeddingModel.OLLAMA.value:
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model="nomic-embed-text",  # Or any Ollama-supported embedding model
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    else:
        raise ValueError(f"Unknown embedding model: {model_id}")


def initialize_llm(config: PipelineConfig):
    """
    Initialize the LLM (Language Model) based on the environment setting.

    Returns:
        A chat model instance compatible with LangChain.
    """
    env = config.environment
    print(f"⚙️  Initializing LLM (environment={env})")

    if env == "api":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.api_model_name,
            temperature=config.temperature,
        )
    elif env in ("local", "hybrid"):
        # 'hybrid' uses local LLM with potentially API embeddings
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=config.llm_model_name,
            temperature=config.temperature,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    else:
        raise ValueError(f"Unknown environment: {env}")


# --- Initialize Models ---
embeddings = initialize_embedding_model(config)
llm = initialize_llm(config)
print("✅ Models initialized successfully.")


# %% [markdown]
# ---
# ## 2. Document Ingestion Pipeline
#
# The ingestion layer converts raw documents into LangChain `Document` objects.
# Each strategy handles different document complexities:
#
# | Strategy | Best For | Handles Tables | OCR | Speed |
# |---|---|---|---|---|
# | `text_loader` | Plain `.txt` files | ❌ | ❌ | ⚡⚡⚡ |
# | `pypdf` | Simple PDFs | ❌ | ❌ | ⚡⚡⚡ |
# | `pymupdf` | Better PDF fidelity | ⚠️ Basic | ❌ | ⚡⚡⚡ |
# | `pdfplumber` | PDFs with tables | ✅ | ❌ | ⚡⚡ |
# | `unstructured` | Mixed content docs | ✅ | ❌ | ⚡⚡ |
# | `unstructured_hires` | Scanned/complex docs | ✅ | ✅ | ⚡ |
# | `docx` | Word documents | ⚠️ | ❌ | ⚡⚡⚡ |
# | `csv` | Tabular data | ✅ | ❌ | ⚡⚡⚡ |
# | `web` | Online documentation | ❌ | ❌ | ⚡⚡ |
# | `directory` | Batch file processing | Depends | Depends | Varies |
# | `llama_parse` | Complex engineering PDFs | ✅ | ✅ | ⚡ |

# %%
# ============================================================
# CELL 3: INGESTION PIPELINE
# ============================================================

def ingest_documents(file_path: str, strategy: str, **kwargs) -> list:
    """
    Universal document ingestion function.

    Args:
        file_path: Path to file or directory.
        strategy: One of IngestionStrategy values.
        **kwargs: Additional loader-specific arguments.

    Returns:
        List of LangChain Document objects.
    """
    print(f"📥 Ingesting '{file_path}' using strategy: {strategy}")

    # --- 1. Plain Text (Most Naive) ---
    if strategy == IngestionStrategy.TEXT_LOADER.value:
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file_path, encoding=kwargs.get("encoding", "utf-8"))
        docs = loader.load()

    # --- 2. PyPDF (Naive PDF) ---
    elif strategy == IngestionStrategy.PYPDF.value:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
        docs = loader.load()

    # --- 3. PyMuPDF (Better PDF Fidelity) ---
    elif strategy == IngestionStrategy.PYMUPDF.value:
        from langchain_community.document_loaders import PyMuPDFLoader
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()

    # --- 4. PDFPlumber (Table-Aware PDF) ---
    elif strategy == IngestionStrategy.PDFPLUMBER.value:
        from langchain_community.document_loaders import PDFPlumberLoader
        loader = PDFPlumberLoader(file_path)
        docs = loader.load()

    # --- 5. Unstructured (Layout-Aware, Mixed Content) ---
    elif strategy == IngestionStrategy.UNSTRUCTURED.value:
        from langchain_community.document_loaders import UnstructuredPDFLoader
        loader = UnstructuredPDFLoader(
            file_path,
            mode=kwargs.get("mode", "elements"),  # "elements" preserves structure
        )
        docs = loader.load()

    # --- 6. Unstructured Hi-Res (OCR + Layout Model) ---
    elif strategy == IngestionStrategy.UNSTRUCTURED_HIRES.value:
        from langchain_community.document_loaders import UnstructuredPDFLoader
        loader = UnstructuredPDFLoader(
            file_path,
            mode="elements",
            strategy="hi_res",            # Uses layout detection model
            infer_table_structure=True,   # Extract tables as structured HTML
            languages=kwargs.get("languages", ["eng"]),
        )
        docs = loader.load()

    # --- 7. Word Documents ---
    elif strategy == IngestionStrategy.DOCX.value:
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file_path)
        docs = loader.load()

    # --- 8. CSV / Structured Data ---
    elif strategy == IngestionStrategy.CSV.value:
        from langchain_community.document_loaders import CSVLoader
        loader = CSVLoader(
            file_path,
            csv_args=kwargs.get("csv_args", {}),
            source_column=kwargs.get("source_column", None),
        )
        docs = loader.load()

    # --- 9. Web Pages ---
    elif strategy == IngestionStrategy.WEB.value:
        from langchain_community.document_loaders import WebBaseLoader
        import bs4
        urls = kwargs.get("urls", [file_path])
        loader = WebBaseLoader(
            web_paths=urls,
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    kwargs.get("html_tags", ["article", "main", "div"])
                )
            ),
        )
        docs = loader.load()

    # --- 10. Directory (Batch Processing) ---
    elif strategy == IngestionStrategy.DIRECTORY.value:
        from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
        glob_pattern = kwargs.get("glob", "**/*.pdf")
        inner_loader_cls = kwargs.get("loader_cls", PyPDFLoader)
        loader = DirectoryLoader(
            file_path,
            glob=glob_pattern,
            loader_cls=inner_loader_cls,
            show_progress=True,
            use_multithreading=True,
            max_concurrency=kwargs.get("max_concurrency", 4),
        )
        docs = loader.load()

    # --- 11. LlamaParse (Vision-AI, Most Advanced) ---
    elif strategy == IngestionStrategy.LLAMA_PARSE.value:
        try:
            from llama_parse import LlamaParse
            from langchain_community.document_loaders import LlamaParsePDFLoader
        except ImportError:
            raise ImportError(
                "LlamaParse requires: pip install llama-parse llama-index\n"
                "Also set LLAMA_CLOUD_API_KEY in your .env file."
            )
        # LlamaParse excels at complex engineering PDFs with schematics,
        # nested tables, multi-column layouts, and embedded figures.
        parser = LlamaParse(
            result_type=kwargs.get("result_type", "markdown"),
            parsing_instruction=kwargs.get(
                "parsing_instruction",
                "This is a technical engineering document. Extract all text, "
                "tables, specifications, and figure captions with high fidelity. "
                "Preserve the hierarchical structure of sections and subsections."
            ),
        )
        documents = parser.load_data(file_path)
        # Convert to LangChain Document format
        from langchain_core.documents import Document
        docs = [Document(page_content=d.text, metadata=d.metadata) for d in documents]

    else:
        raise ValueError(
            f"Unknown ingestion strategy: {strategy}\n"
            f"Available: {[s.value for s in IngestionStrategy]}"
        )

    print(f"   ✅ Loaded {len(docs)} document segments.")

    # --- Post-processing: Metadata Enrichment ---
    for i, doc in enumerate(docs):
        doc.metadata["source_file"] = file_path
        doc.metadata["ingestion_strategy"] = strategy
        doc.metadata["segment_index"] = i

    return docs


# --- Quick Ingestion Inspection Utility ---
def inspect_documents(docs, n: int = 3):
    """Preview the first n ingested documents."""
    print(f"\n📋 Document Preview (showing {min(n, len(docs))} of {len(docs)}):\n")
    for i, doc in enumerate(docs[:n]):
        print(f"--- Document {i+1} ---")
        print(f"  Metadata: {doc.metadata}")
        content_preview = doc.page_content[:300].replace("\n", " ")
        print(f"  Content:  {content_preview}...")
        print()


# ==========================================
# EXAMPLE USAGE:
# ==========================================
# docs = ingest_documents("./data/engineering_manual.pdf", strategy="pymupdf")
# inspect_documents(docs)


# %% [markdown]
# ---
# ## 3. Chunking Strategies
#
# Chunking determines how documents are split into manageable pieces for embedding.
# The choice of strategy dramatically impacts retrieval quality.
#
# | Strategy | Approach | Context Preservation | Best For |
# |---|---|---|---|
# | `fixed_character` | Hard char split | ❌ Very Poor | Quick prototyping only |
# | `recursive` | Paragraph→sentence→word | ⚠️ Good | General purpose (default) |
# | `token_based` | Token-count aware | ⚠️ Good | Aligning with model limits |
# | `sentence` | NLTK sentence boundaries | ✅ Very Good | Narrative text |
# | `spacy_sentence` | spaCy NLP pipeline | ✅ Very Good | Complex sentences |
# | `markdown_header` | Split by headers | ✅ Excellent | Markdown/structured docs |
# | `html_header` | Split by HTML tags | ✅ Excellent | HTML documentation |
# | `semantic` | Embedding similarity | ✅ ✅ Excellent | Mixed-topic documents |
# | `parent_document` | Multi-granularity | ✅ ✅ ✅ Best | Complex engineering docs |

# %%
# ============================================================
# CELL 4: CHUNKING STRATEGIES
# ============================================================

def chunk_documents(docs, strategy: str, config: PipelineConfig = config, **kwargs) -> list:
    """
    Universal document chunking function.

    Args:
        docs: List of LangChain Document objects from ingestion.
        strategy: One of ChunkingStrategy values.
        config: PipelineConfig for default parameters.
        **kwargs: Override chunk_size, chunk_overlap, or strategy-specific params.

    Returns:
        List of chunked Document objects.
    """
    chunk_size = kwargs.get("chunk_size", config.chunk_size)
    chunk_overlap = kwargs.get("chunk_overlap", config.chunk_overlap)
    print(f"✂️  Chunking {len(docs)} documents using strategy: {strategy}")
    print(f"   Parameters: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    # --- 1. Fixed Character Split (Most Naive) ---
    if strategy == ChunkingStrategy.FIXED_CHARACTER.value:
        from langchain_text_splitters import CharacterTextSplitter
        splitter = CharacterTextSplitter(
            separator="",  # Pure character count, no intelligent splitting
            chunk_size=chunk_size,
            chunk_overlap=0,  # Naive = no overlap
        )
        chunks = splitter.split_documents(docs)

    # --- 2. Recursive Character Split (Smart Default) ---
    elif strategy == ChunkingStrategy.RECURSIVE.value:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        chunks = splitter.split_documents(docs)

    # --- 3. Token-Based Split (Model-Aware) ---
    elif strategy == ChunkingStrategy.TOKEN_BASED.value:
        from langchain_text_splitters import TokenTextSplitter
        splitter = TokenTextSplitter(
            chunk_size=kwargs.get("token_chunk_size", 512),
            chunk_overlap=kwargs.get("token_chunk_overlap", 50),
            encoding_name=kwargs.get("encoding", "cl100k_base"),  # GPT-4 tokenizer
        )
        chunks = splitter.split_documents(docs)

    # --- 4. NLTK Sentence Split ---
    elif strategy == ChunkingStrategy.SENTENCE.value:
        from langchain_text_splitters import NLTKTextSplitter
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        splitter = NLTKTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = splitter.split_documents(docs)

    # --- 5. spaCy Sentence Split ---
    elif strategy == ChunkingStrategy.SPACY_SENTENCE.value:
        from langchain_text_splitters import SpacyTextSplitter
        splitter = SpacyTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            pipeline=kwargs.get("spacy_model", "en_core_web_sm"),
        )
        chunks = splitter.split_documents(docs)

    # --- 6. Markdown Header Split (Structure-Preserving) ---
    elif strategy == ChunkingStrategy.MARKDOWN_HEADER.value:
        from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
        headers_to_split_on = kwargs.get("headers", [
            ("#",    "Header 1"),
            ("##",   "Header 2"),
            ("###",  "Header 3"),
            ("####", "Header 4"),
        ])
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )
        # First pass: split by headers
        md_chunks = []
        for doc in docs:
            header_splits = md_splitter.split_text(doc.page_content)
            for split in header_splits:
                split.metadata.update(doc.metadata)
            md_chunks.extend(header_splits)

        # Second pass: ensure chunks aren't too large
        char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = char_splitter.split_documents(md_chunks)

    # --- 7. HTML Header Split ---
    elif strategy == ChunkingStrategy.HTML_HEADER.value:
        from langchain_text_splitters import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter
        headers_to_split_on = kwargs.get("html_headers", [
            ("h1", "Header 1"),
            ("h2", "Header 2"),
            ("h3", "Header 3"),
        ])
        html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        html_chunks = []
        for doc in docs:
            splits = html_splitter.split_text(doc.page_content)
            for split in splits:
                split.metadata.update(doc.metadata)
            html_chunks.extend(splits)

        char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = char_splitter.split_documents(html_chunks)

    # --- 8. Semantic Chunking (Embedding-Based) ---
    elif strategy == ChunkingStrategy.SEMANTIC.value:
        from langchain_experimental.text_splitter import SemanticChunker
        # Groups consecutive sentences whose embeddings are similar.
        # breakpoint_threshold_type options:
        #   "percentile"       – split when distance > Nth percentile
        #   "standard_deviation" – split when distance > N std devs above mean
        #   "interquartile"    – split when distance > IQR-based threshold
        #   "gradient"         – split at largest rate of change in distances
        splitter = SemanticChunker(
            embeddings,
            breakpoint_threshold_type=kwargs.get("breakpoint_type", "percentile"),
            breakpoint_threshold_amount=kwargs.get("breakpoint_amount", 95),
        )
        chunks = splitter.split_documents(docs)

    # --- 9. Parent Document Chunking (Multi-Granularity) ---
    elif strategy == ChunkingStrategy.PARENT_DOCUMENT.value:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        # This strategy creates TWO levels of chunks:
        # - PARENT chunks (large, for context) stored in a docstore
        # - CHILD chunks (small, for precise retrieval) stored in vectorstore
        # When a child is retrieved, the full parent is returned for generation.
        #
        # NOTE: This strategy returns the CHILD chunks for embedding.
        # The parent-child linkage is handled in the retrieval stage
        # via ParentDocumentRetriever. See Cell 7.
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=kwargs.get("parent_chunk_size", 2000),
            chunk_overlap=kwargs.get("parent_chunk_overlap", 400),
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=kwargs.get("child_chunk_size", 400),
            chunk_overlap=kwargs.get("child_chunk_overlap", 50),
        )
        # Split into parent chunks first
        parent_chunks = parent_splitter.split_documents(docs)
        # Split parents into children, preserving parent ID in metadata
        child_chunks = []
        for parent_idx, parent in enumerate(parent_chunks):
            children = child_splitter.split_documents([parent])
            for child in children:
                child.metadata["parent_index"] = parent_idx
                child.metadata["parent_content"] = parent.page_content
            child_chunks.extend(children)

        print(f"   📦 Created {len(parent_chunks)} parent and {len(child_chunks)} child chunks.")
        chunks = child_chunks  # Embed the children

    else:
        raise ValueError(
            f"Unknown chunking strategy: {strategy}\n"
            f"Available: {[s.value for s in ChunkingStrategy]}"
        )

    print(f"   ✅ Created {len(chunks)} chunks.")
    return chunks


# --- Chunk Analysis Utility ---
def analyze_chunks(chunks, show_histogram: bool = False):
    """Analyze chunk size distribution."""
    sizes = [len(c.page_content) for c in chunks]
    print(f"\n📊 Chunk Analysis:")
    print(f"   Total chunks : {len(chunks)}")
    print(f"   Avg size     : {sum(sizes)/len(sizes):.0f} chars")
    print(f"   Min size     : {min(sizes)} chars")
    print(f"   Max size     : {max(sizes)} chars")
    print(f"   Median size  : {sorted(sizes)[len(sizes)//2]} chars")

    if show_histogram:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 4))
            plt.hist(sizes, bins=50, color="#4F46E5", alpha=0.8, edgecolor="white")
            plt.xlabel("Chunk Size (characters)")
            plt.ylabel("Count")
            plt.title("Chunk Size Distribution")
            plt.axvline(sum(sizes)/len(sizes), color="red", linestyle="--", label="Mean")
            plt.legend()
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("   (Install matplotlib for histogram visualization)")


# ==========================================
# EXAMPLE USAGE:
# ==========================================
# chunks = chunk_documents(docs, strategy="recursive")
# analyze_chunks(chunks, show_histogram=True)


# %% [markdown]
# ---
# ## 4. Vector Store Construction
#
# | Backend | Type | Persistence | Filtering | Best For |
# |---|---|---|---|---|
# | `faiss` | In-memory + disk | Save/load | ❌ Basic | Fast local prototyping |
# | `chroma` | Client-server | Built-in | ✅ Rich | Local dev with metadata filtering |
# | `qdrant` | Client-server | Built-in | ✅ Rich | Production-grade deployments |

# %%
# ============================================================
# CELL 5: VECTOR STORE CONSTRUCTION
# ============================================================

def build_vector_store(chunks, embeddings_model, backend: str = None, config: PipelineConfig = config, **kwargs):
    """
    Build a vector store from document chunks.

    Args:
        chunks: List of chunked Document objects.
        embeddings_model: Initialized embedding model.
        backend: One of VectorStoreBackend values (defaults to config).
        config: PipelineConfig for default parameters.

    Returns:
        A vector store instance.
    """
    backend = backend or config.vector_store
    persist_dir = kwargs.get("persist_directory", config.persist_directory)
    print(f"🗄️  Building vector store: {backend}")
    print(f"   Embedding {len(chunks)} chunks...")

    # --- FAISS ---
    if backend == VectorStoreBackend.FAISS.value:
        from langchain_community.vectorstores import FAISS
        vectorstore = FAISS.from_documents(chunks, embeddings_model)
        # Optional: persist to disk
        if kwargs.get("persist", False):
            vectorstore.save_local(persist_dir)
            print(f"   💾 Saved FAISS index to {persist_dir}")

    # --- ChromaDB ---
    elif backend == VectorStoreBackend.CHROMA.value:
        from langchain_community.vectorstores import Chroma
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=persist_dir,
            collection_name=kwargs.get("collection_name", "engineering_oracle"),
        )
        print(f"   💾 ChromaDB persisted to {persist_dir}")

    # --- Qdrant ---
    elif backend == VectorStoreBackend.QDRANT.value:
        from langchain_community.vectorstores import Qdrant
        vectorstore = Qdrant.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            location=kwargs.get("location", ":memory:"),  # In-memory; use URL for server
            collection_name=kwargs.get("collection_name", "engineering_oracle"),
        )

    else:
        raise ValueError(
            f"Unknown vector store backend: {backend}\n"
            f"Available: {[v.value for v in VectorStoreBackend]}"
        )

    print(f"   ✅ Vector store built successfully.")
    return vectorstore


# --- Load Existing Vector Store ---
def load_vector_store(backend: str, embeddings_model, persist_dir: str = "./vector_store", **kwargs):
    """Load a previously persisted vector store."""
    print(f"📂 Loading vector store from {persist_dir}...")

    if backend == VectorStoreBackend.FAISS.value:
        from langchain_community.vectorstores import FAISS
        return FAISS.load_local(persist_dir, embeddings_model, allow_dangerous_deserialization=True)

    elif backend == VectorStoreBackend.CHROMA.value:
        from langchain_community.vectorstores import Chroma
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings_model,
            collection_name=kwargs.get("collection_name", "engineering_oracle"),
        )

    elif backend == VectorStoreBackend.QDRANT.value:
        from langchain_community.vectorstores import Qdrant
        from qdrant_client import QdrantClient
        client = QdrantClient(path=persist_dir)
        return Qdrant(
            client=client,
            collection_name=kwargs.get("collection_name", "engineering_oracle"),
            embeddings=embeddings_model,
        )

    else:
        raise ValueError(f"Unknown backend: {backend}")


# ==========================================
# EXAMPLE USAGE:
# ==========================================
# vectorstore = build_vector_store(chunks, embeddings, backend="faiss", persist=True)
# vectorstore = load_vector_store("faiss", embeddings, persist_dir="./vector_store")


# %% [markdown]
# ---
# ## 5. Retrieval Strategies
#
# Retrieval is the **most critical stage** for RAG performance. The right strategy
# depends on the nature of your queries and the structure of your document corpus.
#
# | Strategy | Type | Diversity | Precision | Speed | Complexity |
# |---|---|---|---|---|---|
# | `naive_similarity` | Dense | ❌ Low | ⚠️ Medium | ⚡⚡⚡ | Trivial |
# | `mmr` | Dense | ✅ High | ⚠️ Medium | ⚡⚡⚡ | Low |
# | `bm25` | Sparse | ❌ Low | ✅ High (exact terms) | ⚡⚡⚡ | Low |
# | `hybrid` | Dense+Sparse | ✅ High | ✅ High | ⚡⚡ | Medium |
# | `multi_query` | Dense+LLM | ✅ High | ✅ High | ⚡ | Medium |
# | `hyde` | Dense+LLM | ⚠️ Medium | ✅ ✅ Very High | ⚡ | High |
# | `self_query` | Dense+Metadata | ⚠️ Medium | ✅ ✅ Very High | ⚡⚡ | High |
# | `parent_document` | Dense | ⚠️ Medium | ✅ ✅ Very High | ⚡⚡ | Medium |
# | `contextual_compression` | Dense+LLM | ⚠️ Medium | ✅ ✅ Very High | ⚡ | High |
# | `cross_encoder_rerank` | Dense+Reranker | ✅ High | ✅ ✅ ✅ Best | ⚡ | High |
# | `cohere_rerank` | Dense+API | ✅ High | ✅ ✅ ✅ Best | ⚡ | Medium |
# | `long_context_reorder` | Post-process | ✅ High | ✅ High | ⚡⚡⚡ | Low |

# %%
# ============================================================
# CELL 6: RETRIEVAL STRATEGIES
# ============================================================

def setup_retriever(
    vectorstore,
    strategy: str,
    chunks: list = None,
    llm_model=None,
    embeddings_model=None,
    config: PipelineConfig = config,
    **kwargs,
):
    """
    Configure a retriever using the specified strategy.

    Args:
        vectorstore: Built vector store instance.
        strategy: One of RetrievalStrategy values.
        chunks: Original chunks (needed for BM25/hybrid/parent strategies).
        llm_model: LLM instance (needed for multi_query/hyde/self_query/compression).
        embeddings_model: Embeddings instance (needed for hyde).
        config: PipelineConfig for default parameters.

    Returns:
        A LangChain retriever instance.
    """
    llm_model = llm_model or llm
    embeddings_model = embeddings_model or embeddings
    top_k = kwargs.get("top_k", config.top_k)
    fetch_k = kwargs.get("fetch_k", config.fetch_k)
    print(f"🔍 Configuring retriever: {strategy} (top_k={top_k})")

    # --- 1. Naive Similarity Search (Baseline) ---
    if strategy == RetrievalStrategy.NAIVE_SIMILARITY.value:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )

    # --- 2. Maximal Marginal Relevance (Diversity) ---
    elif strategy == RetrievalStrategy.MMR.value:
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": top_k,
                "fetch_k": fetch_k,
                "lambda_mult": kwargs.get("lambda_mult", 0.5),  # 0=max diversity, 1=max relevance
            },
        )

    # --- 3. BM25 Keyword Search (Sparse Retrieval) ---
    elif strategy == RetrievalStrategy.BM25.value:
        if chunks is None:
            raise ValueError("BM25 strategy requires the 'chunks' argument.")
        from langchain_community.retrievers import BM25Retriever
        retriever = BM25Retriever.from_documents(chunks)
        retriever.k = top_k

    # --- 4. Hybrid Search (BM25 + Dense Ensemble) ---
    elif strategy == RetrievalStrategy.HYBRID.value:
        if chunks is None:
            raise ValueError("Hybrid strategy requires the 'chunks' argument.")
        from langchain.retrievers import EnsembleRetriever
        from langchain_community.retrievers import BM25Retriever

        dense_retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": top_k}
        )
        sparse_retriever = BM25Retriever.from_documents(chunks)
        sparse_retriever.k = top_k

        weights = kwargs.get("weights", config.hybrid_weights)
        retriever = EnsembleRetriever(
            retrievers=[sparse_retriever, dense_retriever],
            weights=weights,
        )
        print(f"   Ensemble weights: sparse={weights[0]}, dense={weights[1]}")

    # --- 5. Multi-Query Retrieval (LLM Query Expansion) ---
    elif strategy == RetrievalStrategy.MULTI_QUERY.value:
        from langchain.retrievers.multi_query import MultiQueryRetriever
        base_retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": top_k}
        )
        # The LLM generates multiple reformulations of the user's query,
        # retrieves documents for each, and deduplicates the results.
        retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm_model,
        )
        print("   LLM will generate multiple query variants for broader recall.")

    # --- 6. HyDE — Hypothetical Document Embeddings ---
    elif strategy == RetrievalStrategy.HYDE.value:
        from langchain.chains import HypotheticalDocumentEmbedder, LLMChain
        from langchain_core.prompts import ChatPromptTemplate
        # HyDE asks the LLM to generate a hypothetical answer first,
        # embeds that answer, and uses THAT embedding for retrieval.
        # This dramatically improves retrieval for complex queries because
        # the hypothetical document is semantically closer to real docs
        # than the short query itself.
        hyde_prompt = ChatPromptTemplate.from_template(
            "Please write a detailed technical passage that would answer "
            "the following engineering question:\n\n{question}"
        )
        hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
            llm=llm_model,
            base_embeddings=embeddings_model,
            prompt_key="question",
        )
        # Rebuild vector store search with HyDE embeddings
        from langchain_community.vectorstores import FAISS
        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": top_k}
        )
        # Store hyde_embeddings for use during query time
        retriever.search_kwargs["hyde_embeddings"] = hyde_embeddings
        print("   HyDE will generate hypothetical documents before retrieval.")

    # --- 7. Self-Query (Metadata-Aware Structured Retrieval) ---
    elif strategy == RetrievalStrategy.SELF_QUERY.value:
        from langchain.retrievers.self_query.base import SelfQueryRetriever
        from langchain.chains.query_constructor.schema import AttributeInfo
        # Self-query lets the LLM parse the user's question into a
        # structured query with both a semantic search component and
        # metadata filters. Perfect for: "Find specs for grade 316 steel"
        metadata_field_info = kwargs.get("metadata_fields", [
            AttributeInfo(name="source_file", description="The source document filename", type="string"),
            AttributeInfo(name="page", description="The page number in the source document", type="integer"),
            AttributeInfo(name="Header 1", description="The top-level section header", type="string"),
        ])
        document_content_description = kwargs.get(
            "content_description",
            "Technical engineering documentation including specifications, "
            "procedures, material properties, and design guidelines."
        )
        retriever = SelfQueryRetriever.from_llm(
            llm=llm_model,
            vectorstore=vectorstore,
            document_contents=document_content_description,
            metadata_field_info=metadata_field_info,
            enable_limit=True,
        )

    # --- 8. Parent Document Retriever ---
    elif strategy == RetrievalStrategy.PARENT_DOCUMENT.value:
        from langchain.retrievers import ParentDocumentRetriever
        from langchain.storage import InMemoryStore
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # Creates small child chunks for precise retrieval,
        # then returns the larger parent chunk for full context.
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=kwargs.get("parent_size", 2000),
            chunk_overlap=kwargs.get("parent_overlap", 400),
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=kwargs.get("child_size", 400),
            chunk_overlap=kwargs.get("child_overlap", 50),
        )
        docstore = InMemoryStore()
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
        # Add documents to the retriever's internal stores
        if chunks:
            retriever.add_documents(chunks)
            print(f"   Added {len(chunks)} documents to parent-child index.")

    # --- 9. Contextual Compression (LLM-Based) ---
    elif strategy == RetrievalStrategy.CONTEXTUAL_COMPRESSION.value:
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain.retrievers.document_compressors import LLMChainExtractor

        base_retriever = vectorstore.as_retriever(
            search_kwargs={"k": fetch_k}
        )
        # The LLM reads each retrieved chunk and extracts ONLY the parts
        # that are relevant to the query, discarding noise.
        compressor = LLMChainExtractor.from_llm(llm_model)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
        )
        print("   LLM will compress retrieved chunks to only relevant content.")

    # --- 10. Cross-Encoder Re-ranking (Neural Re-scoring) ---
    elif strategy == RetrievalStrategy.CROSS_ENCODER_RERANK.value:
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain.retrievers.document_compressors import CrossEncoderReranker
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder

        # Step 1: Over-retrieve with a base retriever
        base_retriever = vectorstore.as_retriever(
            search_kwargs={"k": fetch_k}  # retrieve more candidates
        )
        # Step 2: Re-score with a Cross-Encoder model
        reranker_model = HuggingFaceCrossEncoder(
            model_name=kwargs.get("reranker_model", "BAAI/bge-reranker-v2-m3")
        )
        compressor = CrossEncoderReranker(
            model=reranker_model,
            top_n=top_k,  # Keep only top_n after re-ranking
        )
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
        )
        print(f"   Cross-Encoder will re-rank {fetch_k} → {top_k} documents.")

    # --- 11. Cohere Re-ranking (API-Based) ---
    elif strategy == RetrievalStrategy.COHERE_RERANK.value:
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain_cohere import CohereRerank

        base_retriever = vectorstore.as_retriever(
            search_kwargs={"k": fetch_k}
        )
        compressor = CohereRerank(
            model="rerank-english-v3.0",
            top_n=top_k,
            cohere_api_key=os.getenv("COHERE_API_KEY"),
        )
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
        )
        print(f"   Cohere will re-rank {fetch_k} → {top_k} documents.")

    # --- 12. Long-Context Reorder ---
    elif strategy == RetrievalStrategy.LONG_CONTEXT_REORDER.value:
        from langchain_community.document_transformers import LongContextReorder
        # Addresses the "lost in the middle" problem: LLMs pay more attention
        # to the beginning and end of context, losing info in the middle.
        # This reorders retrieved docs so the most relevant are at extremes.
        base_retriever = vectorstore.as_retriever(
            search_kwargs={"k": top_k}
        )

        class ReorderingRetriever:
            """Wrapper that applies LongContextReorder after retrieval."""
            def __init__(self, base, reorder_transform):
                self.base = base
                self.reorder = reorder_transform

            def invoke(self, query, **kw):
                docs = self.base.invoke(query, **kw)
                return self.reorder.transform_documents(docs)

            def get_relevant_documents(self, query, **kw):
                return self.invoke(query, **kw)

        reorder = LongContextReorder()
        retriever = ReorderingRetriever(base_retriever, reorder)
        print("   Documents will be reordered to mitigate 'lost-in-the-middle'.")

    else:
        raise ValueError(
            f"Unknown retrieval strategy: {strategy}\n"
            f"Available: {[r.value for r in RetrievalStrategy]}"
        )

    print("   ✅ Retriever configured successfully.")
    return retriever


# --- Retrieval Inspection Utility ---
def inspect_retrieval(retriever, query: str, show_scores: bool = False):
    """Run a test query and inspect retrieved documents."""
    print(f"\n🔎 Query: \"{query}\"\n")
    try:
        docs = retriever.invoke(query)
    except AttributeError:
        docs = retriever.get_relevant_documents(query)

    for i, doc in enumerate(docs):
        print(f"--- Result {i+1} ---")
        if show_scores and "score" in doc.metadata:
            print(f"  Score:    {doc.metadata['score']:.4f}")
        source = doc.metadata.get("source_file", doc.metadata.get("source", "N/A"))
        print(f"  Source:   {source}")
        print(f"  Content:  {doc.page_content[:250].replace(chr(10), ' ')}...")
        print()
    return docs


# ==========================================
# EXAMPLE USAGE:
# ==========================================
# retriever = setup_retriever(vectorstore, strategy="hybrid", chunks=chunks)
# results = inspect_retrieval(retriever, "What is the yield strength of ASTM A36?")


# %% [markdown]
# ---
# ## 6. Generation & Chain Assembly
#
# The generation layer connects retrieved context to the LLM for answer synthesis.
#
# | Mode | Approach | Token Efficiency | Multi-Doc | Memory |
# |---|---|---|---|---|
# | `basic` | Simple RAG chain (LCEL) | ✅ Good | ✅ | ❌ |
# | `stuff` | Stuff all docs into prompt | ❌ May overflow | ✅ | ❌ |
# | `map_reduce` | Process each doc, merge | ✅ Excellent | ✅ | ❌ |
# | `refine` | Iteratively refine answer | ⚠️ High | ✅ | ❌ |
# | `conversational` | Multi-turn with history | ⚠️ Growing | ✅ | ✅ |

# %%
# ============================================================
# CELL 7: GENERATION & CHAIN ASSEMBLY
# ============================================================

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


# --- Prompt Templates ---

ENGINEERING_SYSTEM_PROMPT = (
    "You are the Engineering Oracle — a highly precise technical assistant "
    "specialized in engineering documentation, specifications, and standards. "
    "Answer questions using ONLY the retrieved context provided below. "
    "If the context does not contain sufficient information, state that clearly. "
    "Never fabricate specifications, tolerances, or material properties."
)

BASIC_RAG_TEMPLATE = """
{system_prompt}

### Retrieved Context:
{context}

### Question:
{question}

### Instructions:
- Answer with precise technical detail
- Reference specific sections, page numbers, or table entries when available
- If multiple interpretations exist, present all with their sources
- Use bullet points for clarity on multi-part answers

### Answer:
"""

MAP_TEMPLATE = """
Analyze the following excerpt from engineering documentation and extract
all information relevant to the question.

Document excerpt:
{context}

Question: {question}

Relevant information (or "No relevant information found" if none):
"""

REDUCE_TEMPLATE = """
{system_prompt}

The following are extracted relevant passages from multiple engineering documents:

{summaries}

Based on ALL the extracted information above, provide a comprehensive answer to:
{question}

### Answer:
"""

REFINE_TEMPLATE = """
{system_prompt}

You have an existing partial answer to an engineering question:
{existing_answer}

You have new context that may help refine or correct the answer:
{context}

Question: {question}

Refine the existing answer using the new context. If the new context is not
relevant, return the original answer unchanged.

### Refined Answer:
"""

CONVERSATIONAL_TEMPLATE = """
{system_prompt}

### Retrieved Context:
{context}

Given the conversation history and the retrieved context above, answer the
user's latest question. Maintain consistency with previous answers.

### Answer:
"""

QUERY_REWRITE_TEMPLATE = """
Given the following conversation history and the latest user question,
reformulate the question to be a standalone query that captures all
necessary context for document retrieval.

Chat History:
{chat_history}

Latest Question: {input}

Standalone Question:
"""


def format_docs(docs: list) -> str:
    """Format retrieved documents into a single context string with source attribution."""
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source_file", doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "N/A")
        header = doc.metadata.get("Header 1", "")
        formatted.append(
            f"[Source {i+1}: {source} | Page: {page} | Section: {header}]\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)


def build_rag_chain(retriever, llm_model, mode: str = "basic", **kwargs):
    """
    Build a RAG generation chain.

    Args:
        retriever: Configured retriever instance.
        llm_model: Initialized LLM.
        mode: One of GenerationMode values.

    Returns:
        An invokable chain.
    """
    print(f"🔗 Building generation chain: {mode}")

    # --- 1. Basic RAG Chain (LCEL) ---
    if mode == GenerationMode.BASIC.value:
        prompt = ChatPromptTemplate.from_template(BASIC_RAG_TEMPLATE)

        chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
                "system_prompt": lambda _: ENGINEERING_SYSTEM_PROMPT,
            }
            | prompt
            | llm_model
            | StrOutputParser()
        )

    # --- 2. Stuff Chain (All docs in one prompt) ---
    elif mode == GenerationMode.STUFF.value:
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain.chains import create_retrieval_chain

        prompt = ChatPromptTemplate.from_messages([
            ("system", ENGINEERING_SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion: {input}\n\nAnswer:"),
        ])
        combine_chain = create_stuff_documents_chain(llm_model, prompt)
        chain = create_retrieval_chain(retriever, combine_chain)

    # --- 3. Map-Reduce Chain ---
    elif mode == GenerationMode.MAP_REDUCE.value:
        from langchain.chains.combine_documents import create_stuff_documents_chain

        # Map phase: extract relevant info from each document
        map_prompt = ChatPromptTemplate.from_template(MAP_TEMPLATE)
        # Reduce phase: synthesize all extractions
        reduce_prompt = ChatPromptTemplate.from_template(REDUCE_TEMPLATE)

        def map_reduce_chain_fn(query: str):
            # Retrieve documents
            try:
                docs = retriever.invoke(query)
            except AttributeError:
                docs = retriever.get_relevant_documents(query)

            # Map: Process each document individually
            map_chain = map_prompt | llm_model | StrOutputParser()
            summaries = []
            for doc in docs:
                summary = map_chain.invoke({
                    "context": doc.page_content,
                    "question": query,
                })
                summaries.append(summary)

            # Reduce: Combine all summaries
            reduce_chain = reduce_prompt | llm_model | StrOutputParser()
            final_answer = reduce_chain.invoke({
                "system_prompt": ENGINEERING_SYSTEM_PROMPT,
                "summaries": "\n\n".join(summaries),
                "question": query,
            })
            return final_answer

        chain = RunnableLambda(map_reduce_chain_fn)

    # --- 4. Refine Chain ---
    elif mode == GenerationMode.REFINE.value:
        refine_prompt = ChatPromptTemplate.from_template(REFINE_TEMPLATE)
        initial_prompt = ChatPromptTemplate.from_template(BASIC_RAG_TEMPLATE)

        def refine_chain_fn(query: str):
            try:
                docs = retriever.invoke(query)
            except AttributeError:
                docs = retriever.get_relevant_documents(query)

            if not docs:
                return "No relevant documents found."

            # Initial answer from first document
            initial_chain = initial_prompt | llm_model | StrOutputParser()
            answer = initial_chain.invoke({
                "system_prompt": ENGINEERING_SYSTEM_PROMPT,
                "context": format_docs([docs[0]]),
                "question": query,
            })

            # Refine with each subsequent document
            refine_chain = refine_prompt | llm_model | StrOutputParser()
            for doc in docs[1:]:
                answer = refine_chain.invoke({
                    "system_prompt": ENGINEERING_SYSTEM_PROMPT,
                    "existing_answer": answer,
                    "context": doc.page_content,
                    "question": query,
                })

            return answer

        chain = RunnableLambda(refine_chain_fn)

    # --- 5. Conversational RAG (Multi-Turn with Memory) ---
    elif mode == GenerationMode.CONVERSATIONAL.value:
        from langchain_core.chat_history import InMemoryChatMessageHistory
        from langchain_core.runnables.history import RunnableWithMessageHistory

        # Step 1: Query rewriting chain (contextualizes follow-ups)
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a query reformulation assistant."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("system", QUERY_REWRITE_TEMPLATE),
        ])

        # Step 2: Answer generation chain
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", ENGINEERING_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", CONVERSATIONAL_TEMPLATE),
        ])

        # Build the conversational chain
        from langchain.chains import create_history_aware_retriever, create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain

        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given the chat history and the latest question, "
                       "reformulate it as a standalone question."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm_model, retriever, contextualize_prompt,
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", ENGINEERING_SYSTEM_PROMPT + "\n\nContext:\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        qa_chain = create_stuff_documents_chain(llm_model, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        # Session-based message store
        message_stores = {}

        def get_session_history(session_id: str):
            if session_id not in message_stores:
                message_stores[session_id] = InMemoryChatMessageHistory()
            return message_stores[session_id]

        chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        print("   📝 Conversational memory enabled. Use config={'configurable': {'session_id': 'xxx'}}")

    else:
        raise ValueError(
            f"Unknown generation mode: {mode}\n"
            f"Available: {[g.value for g in GenerationMode]}"
        )

    print("   ✅ Chain built successfully.")
    return chain


# --- Query Execution Helpers ---

def execute_query(query: str, chain, verbose: bool = True) -> str:
    """Execute a query against the RAG chain."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"❓ Query: {query}")
        print(f"{'='*60}\n")

    result = chain.invoke(query)

    # Handle different return types
    if isinstance(result, dict):
        answer = result.get("answer", result.get("output", str(result)))
    else:
        answer = str(result)

    if verbose:
        print(f"💡 Answer:\n{answer}\n")
    return answer


def execute_conversational_query(query: str, chain, session_id: str = "default", verbose: bool = True) -> str:
    """Execute a query with conversational memory."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"💬 [{session_id}] Query: {query}")
        print(f"{'='*60}\n")

    result = chain.invoke(
        {"input": query},
        config={"configurable": {"session_id": session_id}},
    )

    answer = result.get("answer", str(result))
    if verbose:
        print(f"💡 Answer:\n{answer}\n")
    return answer


# ==========================================
# EXAMPLE USAGE:
# ==========================================
# chain = build_rag_chain(retriever, llm, mode="basic")
# execute_query("What are the allowable stress values for SA-516 Grade 70?", chain)
#
# conv_chain = build_rag_chain(retriever, llm, mode="conversational")
# execute_conversational_query("What is the design pressure?", conv_chain, session_id="eng-1")
# execute_conversational_query("And the temperature?", conv_chain, session_id="eng-1")


# %% [markdown]
# ---
# ## 7. Advanced RAG Patterns
#
# These patterns go beyond standard retrieve-and-generate to handle
# challenging real-world engineering query scenarios.

# %%
# ============================================================
# CELL 8: ADVANCED RAG PATTERNS
# ============================================================

# --- 8A. Corrective RAG (CRAG) ---
# If retrieved documents are irrelevant, fall back to web search or
# rephrase the query before generating.

def corrective_rag(query: str, retriever, llm_model, vectorstore, threshold: float = 0.5):
    """
    Corrective RAG: Evaluates retrieval quality and self-corrects.

    Flow:
    1. Retrieve documents normally
    2. LLM grades each document for relevance
    3. If most docs are irrelevant:
       a. Rewrite the query
       b. Re-retrieve with rewritten query
    4. Generate answer from filtered relevant documents
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    print(f"🔧 Corrective RAG — Query: {query}")

    # Step 1: Initial retrieval
    try:
        docs = retriever.invoke(query)
    except AttributeError:
        docs = retriever.get_relevant_documents(query)

    # Step 2: Grade each document
    grading_prompt = ChatPromptTemplate.from_template(
        "You are a grader assessing whether a document is relevant to a question.\n"
        "Document:\n{document}\n\n"
        "Question: {question}\n\n"
        "Answer ONLY 'relevant' or 'irrelevant':"
    )
    grader = grading_prompt | llm_model | StrOutputParser()

    relevant_docs = []
    for doc in docs:
        grade = grader.invoke({
            "document": doc.page_content[:500],
            "question": query,
        }).strip().lower()
        if "relevant" in grade and "irrelevant" not in grade:
            relevant_docs.append(doc)

    relevance_ratio = len(relevant_docs) / max(len(docs), 1)
    print(f"   Relevance ratio: {relevance_ratio:.1%} ({len(relevant_docs)}/{len(docs)} docs)")

    # Step 3: If poor retrieval, rewrite query and retry
    if relevance_ratio < threshold:
        print("   ⚠️  Low relevance detected — rewriting query...")
        rewrite_prompt = ChatPromptTemplate.from_template(
            "The following question did not retrieve good results from engineering docs.\n"
            "Rewrite it to be more specific and use precise engineering terminology.\n\n"
            "Original: {question}\n\nRewritten:"
        )
        rewriter = rewrite_prompt | llm_model | StrOutputParser()
        new_query = rewriter.invoke({"question": query}).strip()
        print(f"   📝 Rewritten query: {new_query}")

        try:
            docs = retriever.invoke(new_query)
        except AttributeError:
            docs = retriever.get_relevant_documents(new_query)
        relevant_docs = docs  # Trust the rewritten query

    # Step 4: Generate with filtered documents
    if not relevant_docs:
        return "No relevant information found in the engineering documentation for this query."

    context = format_docs(relevant_docs)
    gen_prompt = ChatPromptTemplate.from_template(BASIC_RAG_TEMPLATE)
    gen_chain = gen_prompt | llm_model | StrOutputParser()
    answer = gen_chain.invoke({
        "system_prompt": ENGINEERING_SYSTEM_PROMPT,
        "context": context,
        "question": query,
    })
    print(f"\n💡 Answer:\n{answer}")
    return answer


# --- 8B. Adaptive RAG ---
# Routes queries to different retrieval strategies based on query complexity.

def adaptive_rag(query: str, retriever_configs: dict, vectorstore, chunks, llm_model):
    """
    Adaptive RAG: Routes to the optimal retrieval strategy per query.

    Args:
        query: User question.
        retriever_configs: Dict mapping complexity levels to retrieval strategies.
            Example: {"simple": "naive_similarity", "moderate": "hybrid", "complex": "cross_encoder_rerank"}
        vectorstore: Built vector store.
        chunks: Original chunks.
        llm_model: LLM instance.
    """
    print(f"🧠 Adaptive RAG — Classifying query complexity...")

    # Step 1: Classify query complexity
    classify_prompt = ChatPromptTemplate.from_template(
        "Classify the complexity of this engineering question.\n\n"
        "Question: {question}\n\n"
        "Categories:\n"
        "- 'simple': Direct lookup (single fact, specific value)\n"
        "- 'moderate': Requires synthesis across a few sections\n"
        "- 'complex': Requires deep cross-referencing, comparison, or reasoning\n\n"
        "Answer with ONLY one word (simple/moderate/complex):"
    )
    classifier = classify_prompt | llm_model | StrOutputParser()
    complexity = classifier.invoke({"question": query}).strip().lower()

    # Normalize classification
    if "simple" in complexity:
        complexity = "simple"
    elif "moderate" in complexity:
        complexity = "moderate"
    else:
        complexity = "complex"

    strategy = retriever_configs.get(complexity, "hybrid")
    print(f"   Classified as: {complexity.upper()} → Using strategy: {strategy}")

    # Step 2: Build and use the appropriate retriever
    retriever = setup_retriever(
        vectorstore, strategy=strategy, chunks=chunks, llm_model=llm_model
    )

    # Step 3: Generate
    chain = build_rag_chain(retriever, llm_model, mode="basic")
    return execute_query(query, chain)


# --- 8C. Step-Back Prompting RAG ---
# Generates a more abstract "step-back" question for better retrieval.

def stepback_rag(query: str, retriever, llm_model):
    """
    Step-Back RAG: Generates a broader abstraction of the query for retrieval,
    then combines both the specific and abstract contexts.
    """
    print(f"🔙 Step-Back RAG — Query: {query}")

    stepback_prompt = ChatPromptTemplate.from_template(
        "You are an engineering expert. Given a specific technical question, "
        "generate a broader 'step-back' question that covers the underlying "
        "principles or general topic.\n\n"
        "Specific question: {question}\n\n"
        "Step-back question:"
    )
    stepback_chain = stepback_prompt | llm_model | StrOutputParser()
    stepback_query = stepback_chain.invoke({"question": query}).strip()
    print(f"   Step-back question: {stepback_query}")

    # Retrieve for both the original and step-back question
    try:
        specific_docs = retriever.invoke(query)
        broad_docs = retriever.invoke(stepback_query)
    except AttributeError:
        specific_docs = retriever.get_relevant_documents(query)
        broad_docs = retriever.get_relevant_documents(stepback_query)

    # Combine and deduplicate
    seen = set()
    combined_docs = []
    for doc in specific_docs + broad_docs:
        content_hash = hash(doc.page_content[:200])
        if content_hash not in seen:
            seen.add(content_hash)
            combined_docs.append(doc)

    context = format_docs(combined_docs)
    gen_prompt = ChatPromptTemplate.from_template(BASIC_RAG_TEMPLATE)
    gen_chain = gen_prompt | llm_model | StrOutputParser()
    answer = gen_chain.invoke({
        "system_prompt": ENGINEERING_SYSTEM_PROMPT,
        "context": context,
        "question": query,
    })
    print(f"\n💡 Answer:\n{answer}")
    return answer


# ==========================================
# EXAMPLE USAGE:
# ==========================================
# corrective_rag("tensile strength of 304L at 800°F", retriever, llm, vectorstore)
#
# adaptive_rag(
#     "What is the yield stress of SA-516 Gr 70?",
#     retriever_configs={"simple": "naive_similarity", "moderate": "hybrid", "complex": "cross_encoder_rerank"},
#     vectorstore=vectorstore, chunks=chunks, llm_model=llm,
# )
#
# stepback_rag("What is the MDMT for a vessel with 0.5in SA-516-70 shell?", retriever, llm)


# %% [markdown]
# ---
# ## 8. Evaluation Framework
#
# Rigorous evaluation is essential for comparing strategies. This section provides
# both automated metrics (RAGAS) and custom evaluation functions.
#
# ### Metrics Overview
#
# | Metric | Measures | Range | Needs Ground Truth |
# |---|---|---|---|
# | **Faithfulness** | Answer grounded in context | 0–1 | ❌ |
# | **Answer Relevancy** | Answer addresses the question | 0–1 | ❌ |
# | **Context Precision** | Retrieved context contains answer | 0–1 | ✅ |
# | **Context Recall** | All needed info is retrieved | 0–1 | ✅ |
# | **Answer Correctness** | Answer matches ground truth | 0–1 | ✅ |
# | **Retrieval Latency** | Time to retrieve documents | seconds | ❌ |
# | **Chunk Utilization** | % of retrieved context used | 0–1 | ❌ |

# %%
# ============================================================
# CELL 9: EVALUATION FRAMEWORK
# ============================================================

import time
from dataclasses import dataclass


@dataclass
class EvalResult:
    """Container for evaluation results of a single query."""
    query: str
    answer: str
    contexts: list
    ground_truth: str = ""
    retrieval_latency: float = 0.0
    generation_latency: float = 0.0
    num_chunks_retrieved: int = 0
    metrics: dict = field(default_factory=dict)


def evaluate_retrieval_latency(retriever, query: str) -> tuple:
    """Measure retrieval speed and return docs with timing."""
    start = time.perf_counter()
    try:
        docs = retriever.invoke(query)
    except AttributeError:
        docs = retriever.get_relevant_documents(query)
    latency = time.perf_counter() - start
    return docs, latency


def evaluate_generation_latency(chain, query: str) -> tuple:
    """Measure end-to-end generation speed."""
    start = time.perf_counter()
    result = chain.invoke(query)
    latency = time.perf_counter() - start
    answer = result if isinstance(result, str) else result.get("answer", str(result))
    return answer, latency


def evaluate_chunk_utilization(answer: str, contexts: list) -> float:
    """
    Heuristic: What fraction of retrieved chunks contributed to the answer?
    Measured by checking if key phrases from each chunk appear in the answer.
    """
    if not contexts:
        return 0.0
    utilized = 0
    for ctx in contexts:
        # Extract key phrases (first 3 significant words per sentence)
        sentences = ctx.page_content.split(".")
        for sent in sentences[:3]:
            words = [w.strip().lower() for w in sent.split() if len(w.strip()) > 4]
            if any(w in answer.lower() for w in words[:3]):
                utilized += 1
                break
    return utilized / len(contexts)


def run_evaluation(
    retriever,
    chain,
    test_queries: list,
    ground_truths: list = None,
    verbose: bool = True,
) -> list:
    """
    Run evaluation across multiple test queries.

    Args:
        retriever: Configured retriever.
        chain: Built RAG chain.
        test_queries: List of test questions.
        ground_truths: Optional list of expected answers (same length as queries).
        verbose: Print results.

    Returns:
        List of EvalResult objects.
    """
    ground_truths = ground_truths or [""] * len(test_queries)
    results = []

    print(f"\n{'='*60}")
    print(f"📊 EVALUATION — Running {len(test_queries)} queries")
    print(f"{'='*60}\n")

    for i, (query, gt) in enumerate(zip(test_queries, ground_truths)):
        print(f"[{i+1}/{len(test_queries)}] {query}")

        # Retrieval
        docs, ret_latency = evaluate_retrieval_latency(retriever, query)

        # Generation
        answer, gen_latency = evaluate_generation_latency(chain, query)

        # Chunk utilization
        utilization = evaluate_chunk_utilization(answer, docs)

        result = EvalResult(
            query=query,
            answer=answer,
            contexts=[d.page_content for d in docs],
            ground_truth=gt,
            retrieval_latency=ret_latency,
            generation_latency=gen_latency,
            num_chunks_retrieved=len(docs),
            metrics={
                "chunk_utilization": utilization,
                "total_latency": ret_latency + gen_latency,
            },
        )
        results.append(result)

        if verbose:
            print(f"  ⏱️  Retrieval: {ret_latency:.3f}s | Generation: {gen_latency:.3f}s")
            print(f"  📦 Chunks: {len(docs)} | Utilization: {utilization:.1%}")
            print(f"  💡 {answer[:150]}...")
            print()

    # Summary
    avg_ret = sum(r.retrieval_latency for r in results) / len(results)
    avg_gen = sum(r.generation_latency for r in results) / len(results)
    avg_util = sum(r.metrics["chunk_utilization"] for r in results) / len(results)
    print(f"\n{'='*60}")
    print(f"📊 SUMMARY")
    print(f"  Avg Retrieval Latency  : {avg_ret:.3f}s")
    print(f"  Avg Generation Latency : {avg_gen:.3f}s")
    print(f"  Avg Total Latency      : {avg_ret + avg_gen:.3f}s")
    print(f"  Avg Chunk Utilization  : {avg_util:.1%}")
    print(f"{'='*60}\n")

    return results


# --- RAGAS Integration ---
def run_ragas_evaluation(results: list, embeddings_model=None, llm_model=None):
    """
    Run RAGAS evaluation on collected results.
    Requires: pip install ragas

    RAGAS Metrics:
    - faithfulness: Is the answer grounded in the retrieved context?
    - answer_relevancy: Does the answer address the question?
    - context_precision: Are the relevant docs ranked higher?
    - context_recall: Were all necessary pieces of info retrieved?
    """
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset
    except ImportError:
        print("❌ RAGAS not installed. Run: pip install ragas datasets")
        return None

    print("📊 Running RAGAS Evaluation...")

    # Prepare dataset in RAGAS format
    data = {
        "question": [r.query for r in results],
        "answer": [r.answer for r in results],
        "contexts": [r.contexts for r in results],
        "ground_truth": [r.ground_truth for r in results],
    }
    dataset = Dataset.from_dict(data)

    # Select metrics based on whether ground truth is available
    has_ground_truth = any(r.ground_truth for r in results)
    metrics = [faithfulness, answer_relevancy]
    if has_ground_truth:
        metrics.extend([context_precision, context_recall])

    ragas_result = ragas_evaluate(
        dataset,
        metrics=metrics,
        llm=llm_model,
        embeddings=embeddings_model,
    )

    print(f"\n📊 RAGAS Results:")
    for metric, value in ragas_result.items():
        if isinstance(value, (int, float)):
            print(f"   {metric}: {value:.4f}")

    return ragas_result


# --- Strategy Comparison ---
def compare_strategies(
    vectorstore,
    chunks,
    test_queries: list,
    strategies: list,
    llm_model=None,
    embeddings_model=None,
):
    """
    Compare multiple retrieval strategies head-to-head.

    Args:
        vectorstore: Built vector store.
        chunks: Original chunks.
        test_queries: Test questions.
        strategies: List of RetrievalStrategy values to compare.
        llm_model: LLM instance.
        embeddings_model: Embeddings instance.

    Returns:
        Dict mapping strategy → evaluation results.
    """
    llm_model = llm_model or llm
    embeddings_model = embeddings_model or embeddings
    comparison = {}

    print(f"\n{'='*60}")
    print(f"🏆 STRATEGY COMPARISON — {len(strategies)} strategies × {len(test_queries)} queries")
    print(f"{'='*60}\n")

    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---\n")
        retriever = setup_retriever(
            vectorstore, strategy=strategy, chunks=chunks,
            llm_model=llm_model, embeddings_model=embeddings_model,
        )
        chain = build_rag_chain(retriever, llm_model, mode="basic")
        results = run_evaluation(retriever, chain, test_queries, verbose=False)
        comparison[strategy] = results

        # Print summary for this strategy
        avg_latency = sum(r.metrics["total_latency"] for r in results) / len(results)
        avg_util = sum(r.metrics["chunk_utilization"] for r in results) / len(results)
        print(f"  Avg Latency: {avg_latency:.3f}s | Avg Utilization: {avg_util:.1%}")

    # Final comparison table
    print(f"\n{'='*60}")
    print(f"🏆 COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"{'Strategy':<30} {'Avg Latency':>12} {'Avg Util':>12}")
    print(f"{'-'*54}")
    for strategy, results in comparison.items():
        avg_lat = sum(r.metrics["total_latency"] for r in results) / len(results)
        avg_ut = sum(r.metrics["chunk_utilization"] for r in results) / len(results)
        print(f"{strategy:<30} {avg_lat:>10.3f}s {avg_ut:>11.1%}")
    print(f"{'='*60}\n")

    return comparison


# ==========================================
# EXAMPLE USAGE:
# ==========================================
# test_queries = [
#     "What is the maximum allowable stress for SA-516 Gr 70 at 650°F?",
#     "What NDE requirements apply to Category B joints in Division 1?",
#     "Explain the difference between joint efficiency factors E and Ej.",
# ]
# ground_truths = [
#     "The maximum allowable stress for SA-516 Gr 70 at 650°F is 16,600 psi.",
#     "Category B joints require full RT or UT per UW-11(a)(5).",
#     "E is the quality factor from material spec, Ej is the joint efficiency from UW-12.",
# ]
#
# results = run_evaluation(retriever, chain, test_queries, ground_truths)
# ragas_results = run_ragas_evaluation(results, embeddings, llm)
#
# compare_strategies(
#     vectorstore, chunks, test_queries,
#     strategies=["naive_similarity", "hybrid", "cross_encoder_rerank"],
# )


# %% [markdown]
# ---
# ## 9. Use Case Demonstrations
#
# Practical engineering scenarios showcasing the Oracle's capabilities.
# Uncomment and run any use case after building your pipeline.

# %%
# ============================================================
# CELL 10: USE CASE DEMONSTRATIONS
# ============================================================

USE_CASES = {
    "specification_lookup": {
        "name": "📋 Specification Lookup",
        "description": "Direct lookup of material properties, dimensions, tolerances.",
        "queries": [
            "What is the maximum tensile strength of ASTM A36 structural steel?",
            "What are the dimensional tolerances for a Schedule 40 NPS 6 pipe?",
            "What is the minimum yield strength of SA-516 Grade 70 plate?",
        ],
    },
    "troubleshooting": {
        "name": "🔧 Troubleshooting Assistant",
        "description": "Diagnose equipment failures and suggest root causes.",
        "queries": [
            "What are the common causes of stress corrosion cracking in austenitic stainless steel?",
            "The heat exchanger is showing a pressure drop higher than design. What should we check?",
            "What causes hydrogen-induced cracking in carbon steel weldments?",
        ],
    },
    "compliance_check": {
        "name": "✅ Compliance & Code Verification",
        "description": "Verify designs against engineering codes and standards.",
        "queries": [
            "Does a Category C joint in a Division 1 vessel require full radiographic examination?",
            "What are the PWHT requirements for carbon steel vessels per ASME Section VIII?",
            "Is impact testing required for SA-516 Gr 70 at -20°F MDMT?",
        ],
    },
    "cross_reference": {
        "name": "🔗 Cross-Reference Analysis",
        "description": "Find connections across multiple sections or documents.",
        "queries": [
            "How do UG-27 and UG-28 interact for a cylindrical shell under both internal and external pressure?",
            "What is the relationship between joint efficiency and radiographic examination requirements?",
            "Compare the MDMT exemption curves for different material groups.",
        ],
    },
    "design_calculation": {
        "name": "📐 Design Calculation Support",
        "description": "Assist with engineering calculations and formulas.",
        "queries": [
            "What formula should I use to calculate the required thickness of a cylindrical shell under internal pressure?",
            "How do I determine the minimum design metal temperature for a vessel?",
            "What is the procedure for calculating nozzle reinforcement per UG-37?",
        ],
    },
}


def run_use_case(
    use_case_key: str,
    chain,
    retriever=None,
    verbose: bool = True,
):
    """
    Run a predefined use case against the pipeline.

    Args:
        use_case_key: Key from USE_CASES dict.
        chain: Built RAG chain.
        retriever: Optional retriever for evaluation metrics.
        verbose: Print detailed output.
    """
    uc = USE_CASES.get(use_case_key)
    if not uc:
        print(f"❌ Unknown use case: {use_case_key}")
        print(f"Available: {list(USE_CASES.keys())}")
        return

    print(f"\n{'='*60}")
    print(f"{uc['name']}")
    print(f"{uc['description']}")
    print(f"{'='*60}\n")

    for i, query in enumerate(uc["queries"]):
        print(f"[Q{i+1}] {query}")
        answer = execute_query(query, chain, verbose=False)
        print(f"[A{i+1}] {answer[:500]}\n")


def run_all_use_cases(chain, retriever=None):
    """Run all predefined use cases."""
    for key in USE_CASES:
        run_use_case(key, chain, retriever)


# ==========================================
# EXAMPLE USAGE:
# ==========================================
# run_use_case("specification_lookup", chain)
# run_use_case("troubleshooting", chain)
# run_all_use_cases(chain)


# %% [markdown]
# ---
# ## 10. Pipeline Orchestrator
#
# One-command runner that wires everything together.
# Configure once in `PipelineConfig`, then call `run_pipeline()`.

# %%
# ============================================================
# CELL 11: PIPELINE ORCHESTRATOR
# ============================================================

def run_pipeline(
    file_path: str,
    query: str = None,
    config: PipelineConfig = config,
    test_queries: list = None,
    ground_truths: list = None,
    verbose: bool = True,
) -> dict:
    """
    End-to-end pipeline orchestrator.

    Runs the full RAG pipeline from document ingestion through answer generation,
    using strategies defined in the PipelineConfig.

    Args:
        file_path: Path to document(s) to ingest.
        query: Single query to run (optional).
        config: Pipeline configuration.
        test_queries: Batch queries for evaluation (optional).
        ground_truths: Expected answers for test_queries (optional).
        verbose: Detailed logging.

    Returns:
        Dict with all pipeline artifacts:
        {
            "docs": ingested documents,
            "chunks": chunked documents,
            "vectorstore": built vector store,
            "retriever": configured retriever,
            "chain": built RAG chain,
            "answer": single query answer (if query provided),
            "eval_results": evaluation results (if test_queries provided),
        }
    """
    print("\n" + "🔮" * 30)
    print("  ENGINEERING ORACLE — FULL PIPELINE EXECUTION")
    print("🔮" * 30 + "\n")

    results = {}

    # Stage 1: Ingestion
    print("━" * 60)
    print("📥 STAGE 1: Document Ingestion")
    print("━" * 60)
    docs = ingest_documents(file_path, strategy=config.ingestion_strategy)
    results["docs"] = docs

    # Stage 2: Chunking
    print("\n" + "━" * 60)
    print("✂️  STAGE 2: Chunking")
    print("━" * 60)
    chunks = chunk_documents(docs, strategy=config.chunking_strategy, config=config)
    results["chunks"] = chunks
    if verbose:
        analyze_chunks(chunks)

    # Stage 3: Embedding + Vector Store
    print("\n" + "━" * 60)
    print("🗄️  STAGE 3: Vector Store Construction")
    print("━" * 60)
    vectorstore = build_vector_store(chunks, embeddings, backend=config.vector_store, config=config)
    results["vectorstore"] = vectorstore

    # Stage 4: Retrieval
    print("\n" + "━" * 60)
    print("🔍 STAGE 4: Retriever Configuration")
    print("━" * 60)
    retriever = setup_retriever(
        vectorstore,
        strategy=config.retrieval_strategy,
        chunks=chunks,
        llm_model=llm,
        embeddings_model=embeddings,
        config=config,
    )
    results["retriever"] = retriever

    # Stage 5: Generation
    print("\n" + "━" * 60)
    print("🔗 STAGE 5: Chain Assembly")
    print("━" * 60)
    chain = build_rag_chain(retriever, llm, mode=config.generation_mode)
    results["chain"] = chain

    # Stage 6: Execute
    if query:
        print("\n" + "━" * 60)
        print("❓ STAGE 6: Query Execution")
        print("━" * 60)
        if config.generation_mode == GenerationMode.CONVERSATIONAL.value:
            answer = execute_conversational_query(query, chain)
        else:
            answer = execute_query(query, chain)
        results["answer"] = answer

    # Stage 7: Evaluation
    if test_queries:
        print("\n" + "━" * 60)
        print("📊 STAGE 7: Evaluation")
        print("━" * 60)
        eval_results = run_evaluation(
            retriever, chain, test_queries, ground_truths, verbose=verbose
        )
        results["eval_results"] = eval_results

    print("\n" + "🔮" * 30)
    print("  PIPELINE EXECUTION COMPLETE")
    print("🔮" * 30 + "\n")

    return results


# --- Experiment Runner ---
def run_experiment(
    file_path: str,
    test_queries: list,
    experiment_configs: list,
    ground_truths: list = None,
) -> dict:
    """
    Run multiple pipeline configurations and compare results.

    Args:
        file_path: Path to document(s).
        test_queries: Test questions.
        experiment_configs: List of PipelineConfig objects to compare.
        ground_truths: Optional expected answers.

    Returns:
        Dict mapping config label → evaluation results.
    """
    all_results = {}

    print(f"\n{'🧪'*30}")
    print(f"  EXPERIMENT: Comparing {len(experiment_configs)} configurations")
    print(f"{'🧪'*30}\n")

    for i, exp_config in enumerate(experiment_configs):
        label = (
            f"{exp_config.chunking_strategy}+"
            f"{exp_config.retrieval_strategy}+"
            f"{exp_config.embedding_model}"
        )
        print(f"\n{'='*60}")
        print(f"  Experiment {i+1}/{len(experiment_configs)}: {label}")
        print(f"{'='*60}")

        pipeline_result = run_pipeline(
            file_path=file_path,
            config=exp_config,
            test_queries=test_queries,
            ground_truths=ground_truths,
            verbose=False,
        )
        all_results[label] = pipeline_result.get("eval_results", [])

    # Comparison Summary
    print(f"\n{'='*60}")
    print(f"🏆 EXPERIMENT COMPARISON")
    print(f"{'='*60}")
    print(f"{'Configuration':<50} {'Avg Latency':>12} {'Avg Util':>10}")
    print(f"{'-'*72}")
    for label, eval_results in all_results.items():
        if eval_results:
            avg_lat = sum(r.metrics["total_latency"] for r in eval_results) / len(eval_results)
            avg_ut = sum(r.metrics["chunk_utilization"] for r in eval_results) / len(eval_results)
            print(f"{label:<50} {avg_lat:>10.3f}s {avg_ut:>9.1%}")
    print(f"{'='*60}\n")

    return all_results


# ==========================================
# EXAMPLE USAGE — FULL PIPELINE:
# ==========================================
#
# # Option A: Single query execution
# result = run_pipeline(
#     file_path="./data/asme_section_viii_div1.pdf",
#     query="What is the maximum allowable stress for SA-516 Gr 70 at 500°F?",
# )
#
# # Option B: Batch evaluation
# result = run_pipeline(
#     file_path="./data/asme_section_viii_div1.pdf",
#     test_queries=[
#         "What is the required shell thickness for 150 psi internal pressure?",
#         "What PWHT is needed for a 2-inch thick carbon steel vessel?",
#     ],
#     ground_truths=[
#         "Calculated per UG-27 using SE/(P + 0.6SE) formula.",
#         "PWHT is mandatory per UCS-56 for thicknesses exceeding 1.5 inches.",
#     ],
# )
#
# # Option C: Strategy comparison experiment
# configs = [
#     PipelineConfig(chunking_strategy="recursive", retrieval_strategy="naive_similarity"),
#     PipelineConfig(chunking_strategy="recursive", retrieval_strategy="hybrid"),
#     PipelineConfig(chunking_strategy="semantic", retrieval_strategy="cross_encoder_rerank"),
# ]
# experiment_results = run_experiment(
#     file_path="./data/asme_section_viii_div1.pdf",
#     test_queries=["What is the min thickness of a hemispherical head?"],
#     experiment_configs=configs,
# )


# %% [markdown]
# ---
# ## Quick-Start Reference Card
#
# ### 1. Minimal Pipeline (3 lines)
# ```python
# config = PipelineConfig(ingestion_strategy="pypdf", retrieval_strategy="hybrid")
# result = run_pipeline("./data/my_manual.pdf", query="What is X?", config=config)
# print(result["answer"])
# ```
#
# ### 2. Strategy Swap
# ```python
# config.chunking_strategy = "semantic"
# config.retrieval_strategy = "cross_encoder_rerank"
# result = run_pipeline("./data/my_manual.pdf", query="What is X?", config=config)
# ```
#
# ### 3. Full Experiment
# ```python
# configs = [
#     PipelineConfig(retrieval_strategy="naive_similarity"),
#     PipelineConfig(retrieval_strategy="hybrid"),
#     PipelineConfig(retrieval_strategy="cross_encoder_rerank"),
# ]
# run_experiment("./data/my_manual.pdf", test_queries=["..."], experiment_configs=configs)
# ```
#
# ### 4. Conversational Mode
# ```python
# config.generation_mode = "conversational"
# result = run_pipeline("./data/my_manual.pdf", config=config)
# chain = result["chain"]
# execute_conversational_query("What is the design pressure?", chain, session_id="s1")
# execute_conversational_query("And the temperature?", chain, session_id="s1")
# ```
#
# ### 5. Advanced RAG Patterns
# ```python
# corrective_rag("query", retriever, llm, vectorstore)
# stepback_rag("query", retriever, llm)
# adaptive_rag("query", {...}, vectorstore, chunks, llm)
# ```
#
# ---
# **Built for the Engineering Oracle Project**
# *Bridging heavy technical manuals and real-time query resolution through AI-native workflows.*
