from datasets import load_dataset
from functools import cache
from html import escape as escape_html
from langcodes import Language
import numpy as np
import torch
from torch import device, dtype, Tensor
from transformers import MBart50Tokenizer, MBartForConditionalGeneration
from typing import Callable, Tuple
from uroman import Uroman

def get_device_and_torch_dtype() -> Tuple[device, dtype]:
    """Get a device and dtype suitable for the runtime environment."""
    return (device('cuda:0'), torch.float16) if torch.cuda.is_available() else (device('cpu'), torch.float32)

romanize: Callable[[str], str] = Uroman().romanize_string

def create_translator() -> Callable[[str, str], str]: # Based on https://huggingface.co/SnypzZz/Llama2-13b-Language-translate
    """Create an English translator function based on K Damodar Hegde’s Llama2-13b-Language-translate model."""
    model_id = 'SnypzZz/Llama2-13b-Language-translate'
    device, torch_dtype = get_device_and_torch_dtype()
    model = MBartForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch_dtype)
    model.to(device)
    tokenizer = MBart50Tokenizer.from_pretrained(model_id, src_lang='en_XX')

    def translate(text: str, target_lang: str) -> str:
        """Translate text to the target language."""
        inputs = tokenizer(text, return_tensors="pt").to(device)
        translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[target_lang])
        translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return translated_text

    return translate

translate = create_translator()
print(translate('Hello. It’s nice to meet you. Welcome to Italy.', 'it_IT'))