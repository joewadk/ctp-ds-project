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


def save_model_locally(model_id: str, save_directory: str):
    """Save the model and tokenizer locally."""
    model = MBartForConditionalGeneration.from_pretrained(model_id)
    tokenizer = MBart50Tokenizer.from_pretrained(model_id)
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"Model and tokenizer saved to {save_directory}")

#download the model for translations. reduces api calls and speeds up the process of translating
model_id = 'SnypzZz/Llama2-13b-Language-translate'
save_directory = './local_model'
save_model_locally(model_id, save_directory)