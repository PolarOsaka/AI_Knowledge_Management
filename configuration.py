from pathlib import Path
import re

# Mendapatkan direktori tempat file configuration.py ini berada
ROOT_DIR = Path(__file__).resolve().parent

# Menyambung path ke folder data
EXCEL_PATH = ROOT_DIR / "data" / "Project_Registry_Dummy.xlsx"
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_PATH = OUTPUT_DIR / "Dummy_Knowledge_Tree.json"

# Kolom di data project register
REQUIRED_COLUMNS = [
    "Project Code",
    "Project Name",
    "Project Type",
    "Practice Area",
    "Start Year",
]

OPTIONAL_COLUMNS = [
    "Client/Non-Client Billable",
    "Client Name",
    "Sharepoint Folder",
    "Status",
]

# Data dokumen index untuk di-crawl
TREE_PATH = ROOT_DIR / "output" / "Dummy_Knowledge_Tree.json"
DOCUMENTS_DIR = ROOT_DIR / "data" / "documents"
OUTPUT_DIR = ROOT_DIR / "output"

INDEX_PATH = OUTPUT_DIR / "document_index.jsonl"
ERROR_PATH = OUTPUT_DIR / "crawl_errors.json"

SUPPORTED_EXTENSIONS = {".pdf", ".pptx", ".docx", ".xlsx"}

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

# BM25 tokenization baseline. We will refine this later.
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+(?:[._/-][a-zA-Z0-9]+)*")

# Search Request in retrieval API
# Mengatur struktur data yang wajib dikirim oleh klien saat melakukan pencarian ke endpoint /search
MAX_LENGTH_SEARCH = 1000
TOP_K_SEARCH = 5