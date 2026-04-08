import os
import fitz
import io
import torch
from config import MODEL_PATH, PROMPT, CROP_MODE
from PIL import Image
from process.image_process import DeepseekOCR2Processor
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from vllm.model_executor.models.registry import ModelRegistry
from vllm import LLM, SamplingParams
from deepseek_ocr2 import DeepseekOCR2ForCausalLM

os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

ModelRegistry.register_model("DeepseekOCR2ForCausalLM", DeepseekOCR2ForCausalLM)

def pdf_page_to_image(pdf_path, page_num, dpi=144):
    pdf_document = fitz.open(pdf_path)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    page = pdf_document[page_num]
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    img_data = pixmap.tobytes("png")
    img = Image.open(io.BytesIO(img_data))
    pdf_document.close()
    return img

print("Loading model...")
llm = LLM(
    model=MODEL_PATH,
    hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]},
    block_size=256,
    enforce_eager=False,
    trust_remote_code=True,
    max_model_len=8192,
    swap_space=0,
    max_num_seqs=2,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    disable_mm_preprocessor_cache=True
)

logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=20, window_size=50, whitelist_token_ids={128821, 128822})]
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    logits_processors=logits_processors,
    skip_special_tokens=False,
    include_stop_str_in_output=True,
)

print("Processing page 0...")
img = pdf_page_to_image("test_pdf/test.pdf", 0)
cache_item = {
    "prompt": PROMPT,
    "multi_modal_data": {"image": DeepseekOCR2Processor().tokenize_with_images(images=[img], bos=True, eos=True, cropping=CROP_MODE)},
}

outputs = llm.generate([cache_item], sampling_params=sampling_params)
output = outputs[0]
text = output.outputs[0].text
reason = output.outputs[0].finish_reason

print(f"Finish reason: {reason}")
print(f"Has EOS token: {'<｜end▁of▁sentence｜>' in text}")
print(f"Text length characters: {len(text)}")
try:
    print(f"Token count: {len(output.outputs[0].token_ids)}")
except:
    pass

with open("page0_out.txt", "w") as f:
    f.write(text)
print("Done.")
