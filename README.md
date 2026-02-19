# ContractExtractor

A FastAPI-based document processing pipeline for extracting and analyzing contract documents.

## Overview

ContractExtractor processes PDF contracts through multiple stages:
- **PDF Parsing**: Uses LlamaParse to extract text and structure from PDFs
- **Structure Extraction**: Identifies sections, clauses, and document hierarchy
- **Chunking**: Splits documents into manageable chunks for analysis
- **Search & Retrieval**: Enables semantic search across processed documents

## Features

- Drag-and-drop PDF upload interface
- Automatic document structure detection
- Section-based chunking with configurable sizes
- Document explorer with search capabilities
- Dark/Light theme support
- RESTful API for programmatic access

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/ContractExtractor.git
cd ContractExtractor
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables by creating a `.env` file:
```
OPENAI_API_KEY=your_openai_api_key
LLAMA_CLOUD_API_KEY=your_llamaparse_api_key
```

## Usage

1. Start the server:
```bash
cd app
python main.py
```

2. Open your browser at http://localhost:8000

3. Upload a PDF contract using the drag-and-drop interface

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web interface |
| GET | `/health` | Health check |
| POST | `/upload/` | Upload and process PDF |
| GET | `/documents/` | List all documents |
| GET | `/documents/{id}` | Get document info |
| GET | `/documents/{id}/chunks` | Get document chunks |
| GET | `/search/documents?query=` | Search documents |
| DELETE | `/documents/{id}` | Delete a document |

## Project Structure

```
ContractExtractor/
├── app/
│   ├── main.py                 # FastAPI application
│   ├── services/
│   │   ├── utils.py            # Helper functions
│   │   ├── llamaparseProcessor.py
│   │   ├── documentIndexingPipeline.py
│   │   ├── orchestrator.py     # Pipeline orchestration
│   │   └── documentExplorer/   # Document access API
│   └── static/
│       ├── index.html          # Web interface
│       └── style.css
├── requirements.txt
└── .env                        # Environment variables (not tracked)
```

## License

MIT
