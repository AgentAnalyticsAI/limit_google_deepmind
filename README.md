# LIMIT Document Retrieval Evaluation Pipeline

A comprehensive benchmarking framework for evaluating document retrieval systems on the LIMIT dataset from Google DeepMind. This project compares **Pinecone** (vector search) and **WaveflowDB** (hybrid search with VQL) using standard information retrieval metrics.

## 📋 Overview

The LIMIT benchmark evaluates document retrieval systems across 10,000 queries with ground-truth relevance mappings. It supports:

- **Multi-system Comparison**: Side-by-side evaluation of Pinecone and WaveflowDB
- **Comprehensive Metrics**: Precision, Recall, F1-score, MRR, and nDCG@k
- **Scalable Processing**: Multiprocessing support for parallel query evaluation
- **Hybrid Filtering**: Optional query transformation for semantic VQL filtering
- **Performance Tracking**: Detailed timing metrics (embedding, query, total)
- **Flexible Configuration**: Environment-based configuration via `.env` file

## 🎯 Key Features

✅ Processes LIMIT dataset (corpus, queries, relevance labels)  
✅ Generates embeddings and uploads to both vector databases  
✅ Evaluates at multiple top-k values (2, 5, 10)  
✅ Tests with/without hybrid semantic filtering  
✅ Parallel query execution for fast benchmarking  
✅ Comprehensive per-query and aggregated metrics  
✅ Detailed execution logs and results export (CSV/Excel)

## 📁 Project Structure

```
limit_google_deepmind/
├── README.md                          # This file
├── Instructions.txt                   # Step-by-step execution guide
├── requirements.txt                   # Python dependencies
├── utils.py                           # Utility functions
├── 1_create_data.py                   # Extract corpus documents
├── 2_query_map.py                     # Generate query-doc mapping
├── 3_prepare_data_id.py               # Data preparation script
├── 4_waveflow_upload.py               # WaveflowDB data upload
├── 5_pinecone_upload.py               # Pinecone data upload & indexing
├── 6_run_pipeline.py                  # Main evaluation pipeline
├── .env                               # Configuration (secrets)
├── source_data/
│   ├── corpus.jsonl                  # Document corpus
│   ├── queries.jsonl                 # Query texts
│   ├── qrels.jsonl                   # Relevance labels
│   └── query_map.csv                 # Generated: query → doc IDs
├── staging_data/                     # Output from step 1
├── processed_data/                   # Output from step 3
├── logs/                             # Execution logs
└── results/                          # Final results & metrics
    ├── waveflow_results_top*.csv
    ├── pinecone_results_top*.csv
    ├── merged_results_top*.csv
    └── all_results.xlsx
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Dependencies: `pandas`, `spacy`, `sentence-transformers`, `pinecone`,`waveflowdb-client`, `python-dotenv`, `PyPDF2`, `python-docx`, `openpyxl`
- API credentials for Pinecone and WaveflowDB (for Steps 4-5)

### Installation

1. **Clone or navigate to the project**

```bash
cd limit_google_deepmind
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. **Set up environment variables**

Create a `.env` file in the root directory:

```env
# Data paths
DATA_DIR_SOURCE=./source_data
SOURCE_CORPUS_FILE=corpus.jsonl
SOURCE_QUERIES_FILE=queries.jsonl
SOURCE_QREL_FILE=qrels.jsonl
DATA_DIR_TARGET=./staging_data
DATA_DIR_FORMATTED=./processed_data
LOGS=logs
QUERY_MAP_SOURCE_FILE=query_map.csv
DELIMITER=xoxo
TYPE=" AND "

# Pinecone configuration
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
BATCH_SIZE=1000

# WaveflowDB configuration
WAVEFLOWDB_API_KEY=your_waveflowdb_api_key
BASE_URL=your_waveflowdb_url
NAMESPACE=your_namespace
USER_ID=your_user_id
SESSION_ID=your_session_id

# Model configuration
MODEL_NAME=all-MiniLM-L6-v2

# Pipeline configuration
TOP_K=10
MAX_WORKERS_QUERY=4
RESULTS_DIR=./results
```

4. **Download the LIMIT dataset**

Get files from [Google DeepMind GitHub](https://github.com/google-deepmind/limit/tree/main/data/limit):

- `corpus.jsonl`, `queries.jsonl`, `qrels.jsonl`

Place them in `source_data/` folder.

## Obtaining API Keys

- **Pinecone:**

  - Visit the Pinecone console at `https://app.pinecone.io` and sign in or create an account.
  - Create a project (or select an existing one) and navigate to the "API Keys" section.
  - Create a new API key, copy the key value and add it to your `.env` as:

    ```env
    PINECONE_API_KEY=your_pinecone_api_key
    PINECONE_INDEX_NAME=your_index_name
    ```

  - Note: Pinecone may provide separate keys for different environments (dev/prod). Keep keys secret.

- **WaveflowDB:**

  - Visit the Pinecone console at `https://db.agentanalytics.ai`and sign in or create an account.
  - Create a database and then navigate to "API Endpoints"

- Create a new API key, copy the key value and add it to your `.env` as:

  ```env
  WAVEFLOWDB_API_KEY=your_waveflow_api_key
  BASE_URL=https://your-waveflow-host.example.com
  NAMESPACE=your_namespace(Database)
  USER_ID=your_user_id
  SESSION_ID=your_session_id
  ```

## 📊 Execution Steps

### Step 1: Extract Documents

Processes the corpus JSONL and converts to individual text files.

```bash
python 1_create_data.py
```

**Input**: `source_data/corpus.jsonl`  
**Output**: `staging_data/{doc_id}.txt` files

### Step 2: Generate Query Mapping

Creates CSV mapping queries to relevant documents.

```bash
python 2_query_map.py
```

**Input**: `source_data/{corpus, queries, qrels}.jsonl`  
**Output**: `source_data/query_map.csv`

### Step 3: Prepare Data

Formats and organizes data with query ID linking.

```bash
python 3_prepare_data_id.py
```

**Input**: `staging_data/` + `source_data/query_map.csv`  
**Output**: `processed_data/{doc_id}xoxo{query_id}.txt` files

### Step 4: Upload to WaveflowDB

Uploads processed documents to WaveflowDB instance.

```bash
python 4_waveflow_upload.py
```

**Requirements**: WaveflowDB API credentials in `.env`

### Step 5: Upload to Pinecone

Generates embeddings and uploads to Pinecone index.

```bash
python 5_pinecone_upload.py
```

**Output**: `results/pinecone_upload_logs.xlsx` with batch performance metrics

### Step 6: Run Evaluation Pipeline

Executes main benchmarking pipeline with parallel query evaluation.

```bash
python 6_run_pipeline.py
```

**Output**:

- `results/waveflow_results_top{k}_hybrid{filter}.csv`
- `results/pinecone_results_top{k}_hybrid{filter}.csv`
- `results/merged_results_top{k}_hybrid{filter}.csv`
- `results/all_results.xlsx` (raw + aggregated sheets)
- `results/pipeline.log`

## 📈 Evaluation Metrics

### Per-Query Metrics

- **Precision**: Proportion of retrieved documents that are relevant
- **Recall**: Proportion of relevant documents that are retrieved
- **F1-Score**: Harmonic mean of precision and recall
- **MRR**: Mean Reciprocal Rank (1/rank of first relevant doc)
- **nDCG@k**: Normalized Discounted Cumulative Gain (ranking quality at position)

### Performance Metrics

- **Embedding Time**: Time to generate query embeddings
- **Query Time**: Time to search the vector database
- **Total Time**: Combined embedding + query time

## 📝 Utility Functions

### `extract_keywords(text: str)`

Extracts ranked keywords using SpaCy (priority: PROPN > NOUN). Removes stopwords.

### `convert_to_sql_vql(query: str, type: str)`

Transforms natural language queries to VQL format with keyword pairs for hybrid search.

Example: `"machine learning algorithms"` → `SELECT TOP 10 WHERE query is "..." CONTAINS {machine learning} AND {algorithms}`

### `clean_filename_base(fname: str)`

Normalizes filenames: Unicode→ASCII, lowercase, removes invalid chars, handles Windows reserved names.

### `rename_files_in_folder(folder_path: str)`

Batch renames all files in folder using `clean_filename_base()` for consistency.

### `parse_passage_ids(raw: str)`

Parses document IDs from various formats (brackets, quotes, newlines, etc.).

## ⚙️ Configuration Options

### Top-K Values

Evaluate retrieval quality at different result set sizes:

```env
TOP_K=10  # Test at k=2, 5, 10
```

### Hybrid Filtering

Test with and without semantic + keyword filtering:

```python
HYBRID_FILTER_LIST = [True, False]
```

- **True**: WaveflowDB uses VQL transformation; Pinecone skipped
- **False**: Both systems use semantic search only

### Parallel Processing

Control parallel query evaluation:

```env
MAX_WORKERS_QUERY=4  # Number of processes
```

### Embedding Model

Choose sentence transformer model:

```env
MODEL_NAME=all-MiniLM-L6-v2  # Lightweight, fast
```

## 📤 Results Format

### Raw Results CSV

Per-query evaluation metrics:

```
query_id, query_text, precision, recall, f1, mrr, ndcg,
embedding_time, query_time, total_time, retrieved_docs,
relevant_docs, status, system
```

### Aggregated Results (Excel)

Summary statistics grouped by system, top_k, and hybrid_filter:

```
system, top_k, hybrid_filter, avg_precision, avg_recall,
avg_f1, avg_mrr, avg_ndcg
```

## 🔍 Query Dataset

The LIMIT dataset contains 10,000 diverse queries with ground-truth relevance mappings across domains:

| Category  | Count | Examples                        |
| --------- | ----- | ------------------------------- |
| Medical   | ~2000 | Lab reports, clinical notes     |
| Legal     | ~2500 | Contracts, case laws, statutes  |
| Corporate | ~2000 | Annual reports, policies        |
| Academic  | ~1500 | Research papers, dissertations  |
| Finance   | ~1000 | Financial reports, prospectuses |
| Other     | ~1000 | Literature, technical docs      |

Each query includes:

- **Query ID**: Unique identifier
- **Question**: Natural language query text
- **Doc IDs**: Ground truth relevant documents (space/comma-separated)

## 📋 Log Files

Execution logs saved to `logs/` directory:

- `pipeline.log` - Main pipeline execution logs
- `prepare_data_output.csv` - Data preparation summary

## Dataset & Citation

**LIMIT Dataset** - Google DeepMind Large-scale Information Retrieval Benchmark

Citation:

```bibtex
@misc{limit2024,
  title={LIMIT: A Large-scale Information Retrieval Benchmark},
  author={Google DeepMind},
  year={2024},
  howpublished={https://github.com/google-deepmind/limit}
}
```
