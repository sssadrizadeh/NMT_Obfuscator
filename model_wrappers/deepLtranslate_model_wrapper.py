import requests
import torch
# from textattack.models.wrappers import ModelWrapper
from textattack.models.wrappers.model_wrapper import ModelWrapper
from transformers import MarianMTModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepLTranslateModelWrapper(ModelWrapper):
    """
        Abstracts calls to DeepL API, requires API key, max 500,000 chars/month
    """

    def __init__(self):
        self.model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr").to(device)

    def __call__(self, text_input_list, **kwargs):
        res = []
        for text in text_input_list:
            post = requests.post(
                "https://api-free.deepl.com/v2/translate",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "DeepL-Auth-Key acfda7cb-1667-a5fd-e763-85f96e107d76:fx"
                },
                json={
                    "text": [text],
                    "source_lang": "EN",
                    "target_lang": "FR"
                }
            )
            res.append(post.json()["translations"][0]["text"])
        return res
