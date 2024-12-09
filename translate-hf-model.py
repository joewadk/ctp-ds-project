def translate_hf_model(text, target_lang):
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
        return (device('cuda:0'), torch.float16) if torch.cuda.is_available() else (device('cpu'), torch.float32) #hoepfully uses gpu if available

    romanize: Callable[[str], str] = Uroman().romanize_string #converts text to romanized text

    def create_translator(local_model_path: str) -> Callable[[str, str], str]: #creates a translator function
        """Create an English translator function using a locally saved model."""
        device, torch_dtype = get_device_and_torch_dtype()
        model = MBartForConditionalGeneration.from_pretrained(local_model_path, torch_dtype=torch_dtype)
        model.to(device)
        tokenizer = MBart50Tokenizer.from_pretrained(local_model_path, src_lang='en_XX')

        def translate(text: str, target_lang: str) -> str: #translates text to target language
            """Translate text to the target language."""
            inputs = tokenizer(text, return_tensors="pt").to(device)
            translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[target_lang])
            translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            return translated_text

        return translate

    #calling the local model
    local_model_path = './local_model'
    translate = create_translator(local_model_path)
    return(translate(text, target_lang))
print(translate_hf_model('Hello.', 'it_IT'))