import functools

import torch
from textattack.models.wrappers import HuggingFaceModelWrapper
from transformers import MarianMTModel, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MarianMTModelWrapper(HuggingFaceModelWrapper):

    def __init__(self, model_name: str):
        super().__init__(
            MarianMTModel.from_pretrained(model_name).to(device),
            AutoTokenizer.from_pretrained(model_name)
        )

    def __call__(self, text_input_list):
        inputs = self.tokenizer_call(text_input_list)
        return [
            self.tokenizer.decode(ids, skip_special_tokens=True) for ids in
            self.generate(inputs)
        ]

    @functools.lru_cache(maxsize=2 ** 12)
    def generate(self, inputs):
        return self.model.generate(inputs)

    def tokenizer_call(self, text_input_list):
        inputs = self.tokenizer(
            text_input_list,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )["input_ids"]
        return inputs.to(device)