import os

# Prevent thread thrashing when using multiprocessing/threading with PyTorch and numpy
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv

# Load environment variables from .env file (if present)
load_dotenv()

BASE_SIZE = int(os.getenv('BASE_SIZE', '1024'))
IMAGE_SIZE = int(os.getenv('IMAGE_SIZE', '768'))
CROP_MODE = os.getenv('CROP_MODE', 'True').lower() in ('true', '1', 't')
MIN_CROPS = int(os.getenv('MIN_CROPS', '2'))
MAX_CROPS = int(os.getenv('MAX_CROPS', '6')) # max:6
MAX_CONCURRENCY = int(os.getenv('MAX_CONCURRENCY', '100')) # If you have limited GPU memory, lower the concurrency count.
NUM_WORKERS = int(os.getenv('NUM_WORKERS', '64')) # image pre-process (resize/padding) workers
PDF_BATCH_SIZE = int(os.getenv('PDF_BATCH_SIZE', '12'))  # Process PDF pages in batches. Lower batch size allows interleaving concurrent PDF tasks!
MAX_PDF_WORKERS = int(os.getenv('MAX_PDF_WORKERS', '8')) # Number of separate PDFs to process concurrently.
MAX_PAGES = int(os.getenv('MAX_PAGES', '0')) # Maximum number of pages per PDF (0 means unlimited)
ENABLE_CACHE = os.getenv('ENABLE_CACHE', 'False').lower() in ('true', '1', 't') # Disable PDF hash caching by default

# Memory monitoring thresholds (percentage of total system memory)
MEMORY_WARNING_THRESHOLD = float(os.getenv('MEMORY_WARNING_THRESHOLD', '80.0'))   # Log warning when memory usage exceeds this
MEMORY_CRITICAL_THRESHOLD = float(os.getenv('MEMORY_CRITICAL_THRESHOLD', '90.0'))  # Trigger shutdown/restart when memory usage exceeds this
MEMORY_CHECK_INTERVAL = int(os.getenv('MEMORY_CHECK_INTERVAL', '30'))        # Seconds between memory checks

PRINT_NUM_VIS_TOKENS = os.getenv('PRINT_NUM_VIS_TOKENS', 'False').lower() in ('true', '1', 't')
SKIP_REPEAT = os.getenv('SKIP_REPEAT', 'True').lower() in ('true', '1', 't')
MODEL_PATH = os.getenv('MODEL_PATH', 'deepseek-ai/DeepSeek-OCR-2') # change to your model path

# TODO: change INPUT_PATH
# .pdf: run_dpsk_ocr_pdf.py; 
# .jpg, .png, .jpeg: run_dpsk_ocr_image.py; 
# Omnidocbench images path: run_dpsk_ocr_eval_batch.py

INPUT_PATH = os.getenv('INPUT_PATH', '~/input_pdf')
OUTPUT_PATH = os.getenv('OUTPUT_PATH', '~/output')

PROMPT = os.getenv('PROMPT', '<image>\n<|grounding|>Convert the document to markdown.')
# PROMPT = '<image>\nFree OCR.'
# .......

from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
