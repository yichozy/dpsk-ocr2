
BASE_SIZE = 1024
IMAGE_SIZE = 768
CROP_MODE = True
MIN_CROPS= 2
MAX_CROPS= 6 # max:6
MAX_CONCURRENCY = 100 # If you have limited GPU memory, lower the concurrency count.
NUM_WORKERS = 64 # image pre-process (resize/padding) workers
PDF_BATCH_SIZE = 4  # Process PDF pages in batches to limit system memory usage. Lower if you have limited RAM.

# Memory monitoring thresholds (percentage of total system memory)
MEMORY_WARNING_THRESHOLD = 80.0   # Log warning when memory usage exceeds this
MEMORY_CRITICAL_THRESHOLD = 90.0  # Trigger shutdown/restart when memory usage exceeds this
MEMORY_CHECK_INTERVAL = 30        # Seconds between memory checks

PRINT_NUM_VIS_TOKENS = False
SKIP_REPEAT = True
MODEL_PATH = 'deepseek-ai/DeepSeek-OCR-2' # change to your model path

# TODO: change INPUT_PATH
# .pdf: run_dpsk_ocr_pdf.py; 
# .jpg, .png, .jpeg: run_dpsk_ocr_image.py; 
# Omnidocbench images path: run_dpsk_ocr_eval_batch.py



INPUT_PATH = '/your/image/path/'
OUTPUT_PATH = '/your/output/path/'

PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'
# PROMPT = '<image>\nFree OCR.'
# .......


from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
