import torch
from torch import device, dtype
from transformers import MBart50Tokenizer, MBartForConditionalGeneration
import torch.nn.utils.prune as prune
from typing import Callable, Tuple, List, Union
from uroman import Uroman
import os

'''
Methods tried to speed this up: 

Batch Processing [FAILED]
Quantization of Model [FAILED]
Model Pruning (up to 80%) [FAILED]

Conclusion: I cannot get this to run locally on Apple MacBook Pro with M2 Chip with MPS

-Mackenzie 
'''

# use CPU for quantization operation which isn't supported on MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Checking hardware
# print(torch.backends.mps.is_available())
# Displays QNN which is commonly used for mobile devices?
# print(torch.backends.quantized.supported_engines)

# Pruning model reduces accuracy but might increase speed for large models


def prune_model(model, pruning_percentage=0.2):
    """Prune the Linear layers in the model to reduce the number of parameters."""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Apply pruning on the Linear layers by keeping a percentage of weights
            prune.l1_unstructured(module, name="weight",
                                  amount=pruning_percentage)
            print(f"Pruned {name} - {pruning_percentage*100}% of weights.")

    # Optionally remove the pruning reparameterization (prune mask)
    # This will leave the model with a final pruned version (not a reparameterized version).
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.remove(module, 'weight')
            print(f"Removed pruning reparameterization from {name}.")

    return model


# Decides what hardware to use
def get_device_and_torch_dtype() -> Tuple[device, dtype]:
    """Get a device and dtype suitable for the runtime environment."""
    # return (device('cuda:0'), torch.float16) if torch.cuda.is_available() else (device('cpu'), torch.float32)
    if torch.cuda.is_available():
        return (device('cuda:0'), torch.float16)
    elif torch.backends.mps.is_available():
        # for Apple Silicon
        return (device('mps'), torch.float32)
    else:
        return (device('cpu'), torch.float32)


romanize: Callable[[str], str] = Uroman().romanize_string


# Based on https://huggingface.co/SnypzZz/Llama2-13b-Language-translate
def create_translator() -> Callable[[str, str], str]:
    """Create an English translator function based on K Damodar Hegde’s Llama2-13b-Language-translate model."""
    model_id = 'SnypzZz/Llama2-13b-Language-translate'
    device, torch_dtype = get_device_and_torch_dtype()
    model = MBartForConditionalGeneration.from_pretrained(
        # Set CPU memory usage to low
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True)

    '''
    FAILED ATTEMPT AT IMPLEMENTING QUANTIZATION, DIFFICULTIES DUE TO MPS NOT SUPPORTING IT

    torch.backends.quantized.engine = 'qnnpack'
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model.to('cpu')
    # Apply dynamic quantization (for faster inference and lower memory usage)
    model = torch.quantization.quantize_dynamic(
        # Quantizing linear layers to int8
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    '''

    model = model.to(device)

    # Apply pruning to the model to reduce the size of Linear layers
    # 50% is extreme
    model = prune_model(model, pruning_percentage=0.8)

    # Set model_max_length
    tokenizer = MBart50Tokenizer.from_pretrained(
        model_id, src_lang='en_XX', model_max_length=512)
    # print(tokenizer.model_max_length)
    # print(type(tokenizer.model_max_length))

    # Update function argument for batch support, takes a single string or a list of strings
    def translate(texts: Union[str, List[str]], target_lang: str) -> str:
        """Translate text to the target language."""
        # Enabled truncation and padding
        inputs = tokenizer(texts, return_tensors="pt",
                           truncation=True, padding=True).to(device)
        translated_tokens = model.generate(
            **inputs, forced_bos_token_id=tokenizer.lang_code_to_id[target_lang])
        translated_text = tokenizer.batch_decode(
            translated_tokens, skip_special_tokens=True)[0]
        return translated_text

    return translate

# Main translation function that the frontend will invoke


def run_translation(texts: Union[str, List[str]], target_lang: str):
    translate = create_translator()
    return translate(texts, target_lang)

# Testing


def test_translations_for_demo(list_of_gestures_to_test: List[str], target_langs: List[str]):
    result = {}

    for target_lang in target_langs:
        result[target_lang] = run_translation(
            list_of_gestures_to_test, target_lang)

    for key in result:
        print(result[key])


list_of_gestures = ['blow', 'wait', 'cloud', 'bird', 'owie', 'duck', 'minemy', 'lips', 'flower', 'time', 'vacuum', 'apple', 'puzzle', 'mitten', 'there', 'dry', 'shirt', 'owl', 'yellow', 'not', 'zipper', 'clean', 'closet', 'quiet', 'have', 'brother', 'clown', 'cheek', 'cute', 'store', 'shoe', 'wet', 'see', 'empty', 'fall', 'balloon', 'frenchfries', 'finger', 'same', 'cry', 'hungry', 'orange', 'milk', 'go', 'drawer', 'TV', 'another', 'giraffe', 'wake', 'bee', 'bad', 'can', 'say', 'callonphone', 'finish', 'old', 'backyard', 'sick', 'look', 'that', 'black', 'yourself', 'open', 'alligator', 'moon', 'find', 'pizza', 'shhh', 'fast', 'jacket', 'scissors', 'now', 'man', 'sticky', 'jump', 'sleep', 'sun', 'first', 'grass', 'uncle', 'fish', 'cowboy', 'snow', 'dryer', 'green', 'bug', 'nap', 'feet', 'yucky', 'morning', 'sad', 'face', 'penny', 'gift', 'night', 'hair', 'who', 'think', 'brown', 'mad', 'bed', 'drink', 'stay', 'flag', 'tooth', 'awake', 'thankyou', 'hot', 'like', 'where', 'hesheit', 'potty', 'down', 'stuck', 'no', 'head', 'food', 'pretty', 'nuts', 'animal', 'frog', 'beside', 'noisy', 'water', 'weus', 'happy',
                    'white', 'bye', 'high', 'fine', 'boat', 'all', 'tiger', 'pencil', 'sleepy', 'grandma', 'chocolate', 'haveto', 'radio', 'farm', 'any', 'zebra', 'rain', 'toy', 'donkey', 'lion', 'drop', 'many', 'bath', 'aunt', 'will', 'hate', 'on', 'pretend', 'kitty', 'fireman', 'before', 'doll', 'stairs', 'kiss', 'loud', 'hen', 'listen', 'give', 'wolf', 'dad', 'gum', 'hear', 'refrigerator', 'outside', 'cut', 'underwear', 'please', 'child', 'smile', 'pen', 'yesterday', 'horse', 'pig', 'table', 'eye', 'snack', 'story', 'police', 'arm', 'talk', 'grandpa', 'tongue', 'pool', 'girl', 'up', 'better', 'tree', 'dance', 'close', 'taste', 'chin', 'ride', 'because', 'if', 'cat', 'why', 'carrot', 'dog', 'mouse', 'jeans', 'shower', 'later', 'mom', 'nose', 'yes', 'airplane', 'book', 'blue', 'icecream', 'garbage', 'tomorrow', 'red', 'cow', 'person', 'puppy', 'cereal', 'touch', 'mouth', 'boy', 'thirsty', 'make', 'for', 'glasswindow', 'into', 'read', 'every', 'bedroom', 'napkin', 'ear', 'toothbrush', 'home', 'pajamas', 'hello', 'helicopter', 'lamp', 'room', 'dirty', 'chair', 'hat', 'elephant', 'after', 'car', 'hide', 'goose']

list_of_languages_for_demo = [
    'es_XX', 'fr_XX', 'ja_XX', 'de_DE', 'zh_CN', 'hi_IN', 'ar_AR', 'ru_RU', 'ur_PK', 'bn_IN']

test_translations_for_demo(list_of_gestures, list_of_languages_for_demo)
