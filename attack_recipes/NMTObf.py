import functools
import logging
import os
import random
import sys
from contextlib import contextmanager
from timeit import default_timer as timer
from typing import List

import evaluate
import nltk
import torch
from torch.nn import functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from model_wrappers.deepLtranslate_model_wrapper import DeepLTranslateModelWrapper

logging.getLogger("transformers").setLevel(logging.CRITICAL)
DEEPL = DeepLTranslateModelWrapper()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@contextmanager
def suppress_stderr():
    # Too many tqdm's otherwise
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


class NMTObf:

    @staticmethod
    def attack(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, original_texts: List[str],
               text_to_inject: str, log_file: str, iterations=100, neighbours=20):
        AttackSearch(model, tokenizer, original_texts, text_to_inject, log_file, iterations,
                            neighbours).attack_dataset()


class AttackSearch:

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, original_texts: List[str],
                 text_to_inject: str, log_file: str, iterations: int, neighbours: int, only_ascii=True):
        # Initialize the model and tokenizer, set the model to evaluation mode
        self.model_wrapper = WhiteBoxModelWrapper(
            model, tokenizer,
            relevant_token_predicate=lambda x: x.isascii() and x.isalpha() if only_ascii else True
        )

        # Shuffle the dataset
        self.original_texts = iter(original_texts)

        # Initialize current state
        self.initial_text, self.original_translation, self.original_translation_ids = None, None, None
        self.text_to_inject = text_to_inject
        self.lm_wrapper = MaskedLMModelWrapper(text_to_inject)
        self.log_file = log_file
        self.iterations = iterations
        self.neighbours = neighbours

    def attack_dataset(self):
        self.print_and_log("\n ============= STARTING ATTACK: " + (self.text_to_inject) + " ============== \n")
        while True:
            try:
                self.next()
            except StopIteration:
                return
            self.print_and_log(f"Original translation: {self.initial_text} -> {self.original_translation} \n")
            adversarial_sentence, adversarial_translation = self.attack_instance()
            if adversarial_translation is not None:
                self.log_success(adversarial_sentence, adversarial_translation)
            else:
                self.log_failure()

    def attack_instance(self):

        # Init the adversarial state
        original_embeddings = self.model_wrapper.get_input_embeddings(self.initial_text).to(device)
        adversarial_obfuscator_embeddings = self.init_adversarial_obfuscator() \
            .clone().detach().requires_grad_(True).to(device)
        adversarial_target_embeddings = self.model_wrapper.get_input_embeddings(self.text_to_inject).to(device)

        # Init the optimizer and the loss
        optimizer = torch.optim.Adam([adversarial_obfuscator_embeddings], lr=0.04)
        prev_char_id, prev_chars, prev_loss = None, [], 100000

        # Iterate while the obfuscation is not successful
        start = timer()
        for i in range(self.iterations):
            # Get the adversarial embeddings that the concatenation yields:
            # original_embeddings + adversarial_obfuscator_embeddings + adversarial_target_embeddings, i.e.
            # "I have a dog" + (obfuscator) + "I will definitely burn the parliament"
            adversarial_embeddings = torch.cat([
                original_embeddings[:, :-1, :], adversarial_obfuscator_embeddings, adversarial_target_embeddings
            ], dim=1).to(device)

            # Project the obfuscator into interesting tokens
            timer_start = timer()
            obfuscator_projection, obfuscator_token = self.get_optimal_embedding(adversarial_obfuscator_embeddings)
            timer_end = timer()
            print(f"Optimal embedding computation took {timer_end - timer_start} seconds")

            # Get the projected sentence
            timer_start = timer()
            adversarial_embedding_projections = torch.cat([
                original_embeddings[:, :-1, :], obfuscator_projection, adversarial_target_embeddings
            ], dim=1).to(device)
            timer_end = timer()
            print(f"Projection took {timer_end - timer_start} seconds")

            # Get the corresponding token ids
            adversarial_token_ids = self.model_wrapper.get_embeddings_to_token_ids(adversarial_embedding_projections)

            # Get the adversarial sentence resulting from the projection
            adversarial_sentence = self.model_wrapper.tokenizer.decode(adversarial_token_ids, skip_special_tokens=True) \
                .replace("▁", " ")
            print(adversarial_sentence)

            # Get the loss of the adversarial sentence and backpropagate
            timer_start = timer()
            adversarial_loss = self.model_wrapper.call_with_input_embeddings(
                adversarial_embeddings.to(device), self.original_translation_ids.to(device)).loss
            timer_end = timer()
            print(f"Loss computation took {timer_end - timer_start} seconds")

            timer_start = timer()
            optimizer.zero_grad()
            adversarial_loss.backward(retain_graph=True)
            optimizer.step()
            timer_end = timer()
            print(f"Backpropagation took {timer_end - timer_start} seconds")

            # Log the state
            timer_start = timer()
            adversarial_translation = self.model_wrapper.get_translation(adversarial_sentence)
            timer_end = timer()
            print(f"Translation took {timer_end - timer_start} seconds\n")

            new_char_id = self.model_wrapper.get_embedding_to_token_id(obfuscator_projection[0][0]).item()
            if new_char_id != prev_char_id and prev_char_id is not None:
                self.print_and_log(f"Optimal token: {(obfuscator_token)}")
                self.print_and_log(
                    f"{('[Iteration ' + str(i) + ']')} {adversarial_sentence} -> {adversarial_translation} -> {(str(adversarial_loss.item())[:6])}\n")
                prev_chars.append(prev_char_id)

            # If we are stuck, reinitialize the obfuscator
            if torch.abs(prev_loss - adversarial_loss) < 0.01 or new_char_id in prev_chars:
                adversarial_obfuscator_embeddings = self.init_adversarial_obfuscator() \
                    .clone().detach().requires_grad_(True).to(device)
                optimizer = torch.optim.Adam([adversarial_obfuscator_embeddings], lr=0.04)

            # If we are successful, return the adversarial sentence
            if self.is_success(adversarial_sentence, adversarial_translation):
                end = timer()
                self.print_and_log(f"{('[Iteration ' + str(i) + ' (' + str(end - start)[:5] + 's) ]')}")
                return adversarial_sentence, adversarial_translation

            # Update the loss
            prev_loss = adversarial_loss
            prev_char_id = new_char_id

        return None, None

    def get_optimal_embedding(self, adversarial_obfuscator_embeddings):
        projections_by_distance = self.model_wrapper.sort_relevant_internal_embeddings(
            adversarial_obfuscator_embeddings[0, 0, :]
        )
        token_ids = self.model_wrapper.get_embeddings_to_token_ids(projections_by_distance[:, :self.neighbours, :])
        tokens = self.model_wrapper.tokenizer.convert_ids_to_tokens(token_ids)
        tokens_by_perplexity = dict(zip(tokens, self.lm_wrapper.get_perplexities([
            f"{self.initial_text} {token} {self.text_to_inject}" for token in tokens
        ])))
        min_token = min(tokens_by_perplexity.keys(), key=lambda x: tokens_by_perplexity.get(x, 100000))
        return self.model_wrapper.get_input_embeddings(min_token)[:, :-1, :], min_token

    def is_success(self, adversarial_sentence, adversarial_translation):
        return adversarial_translation is not None and \
            nltk.edit_distance(self.original_translation, adversarial_translation) < 7

    def init_adversarial_obfuscator(self):
        return self.model_wrapper.relevant_internal_embeddings[
               random.randint(0, len(self.model_wrapper.relevant_internal_embeddings) - 1), :] \
            .unsqueeze(0).unsqueeze(0)

    def get_adversarial_embeddings(self, original_embeddings, adversarial_obfuscator_embeddings,
                                   adversarial_target_embeddings):
        return torch.cat([
            original_embeddings[:, :-1, :],
            adversarial_obfuscator_embeddings,
            adversarial_target_embeddings,
        ], dim=1)

    def next(self):
        self.initial_text = next(self.original_texts)
        self.original_translation = self.model_wrapper.get_translation(self.initial_text)
        self.original_translation_ids = self.model_wrapper.get_translation_ids(self.initial_text)

    def log_success(self, adversarial_sentence, adversarial_translation):
        self.print_and_log(("\n================= SUCCESS =================\n"))
        self.print_and_log(f"Appending target sentence: {self.initial_text} --> {adversarial_sentence}\n")
        self.print_and_log(f"Original translation: {self.original_translation}")
        self.print_and_log(f"Modified translation: {adversarial_translation}\n")
        self.print_and_log(("===========================================\n"))

    def log_failure(self):
        self.print_and_log(f"================= {('FAIL')} =================\n")

    def print_and_log(self, text):
        print(text)
        path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', self.log_file))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a+") as f:
            f.write(text + "\n")


class MaskedLMModelWrapper:

    def __init__(self, target):
        self.perplexity = evaluate.load("perplexity", module_type="metric", keep_in_memory=True)
        self.cache = {}

    def get_perplexities(self, input_texts):
        logging.getLogger("evaluate").setLevel(logging.CRITICAL)
        remaining = [text for text in input_texts if text not in self.cache]
        with suppress_stderr():
            tmp = dict(zip(remaining, self.perplexity.compute(
                model_id="gpt2",
                add_start_token=False,
                predictions=remaining,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )["perplexities"])) if len(remaining) > 0 else {}
        self.cache.update(tmp)
        return [self.cache[text] for text in input_texts]

class WhiteBoxModelWrapper:

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, relevant_token_predicate=None):
        # Initialize the model and tokenizer
        self.model, self.tokenizer = model, tokenizer

        # Get the full tokenizer vocabulary
        self.vocabulary = tokenizer.get_vocab()

        # Get the internal lookup table for the embeddings
        self.internal_embeddings = torch.stack([
            model.get_encoder().embed_tokens(torch.tensor([i]).to(device)).squeeze(0)
            for i in range(len(self.vocabulary))
        ]).to(device)

        # self.internal_embeddings = model.get_input_embeddings()(torch.tensor(list(self.vocabulary.values())).to(device))
        self.relevant_internal_embeddings = model.get_input_embeddings()(torch.tensor([
            i for i in range(len(self.vocabulary)) if relevant_token_predicate(self.tokenizer.convert_ids_to_tokens(i))
        ]).to(device))

        # Get the embedding scale
        self.embed_scale = model.get_encoder().embed_scale if hasattr(model.get_encoder(), "embed_scale") else 1.0

    def get_translation_ids(self, text):
        return self.model.generate(
            self.tokenizer.encode(text, return_tensors="pt").to(device)
        )

    @functools.lru_cache(maxsize=10000)
    def get_translation(self, text):
        translation_ids = self.get_translation_ids(text)
        return self.tokenizer.decode(translation_ids[0], skip_special_tokens=True) \
            .replace("▁", " ")

    def get_input_embeddings(self, text):
        return self.model.get_encoder().embed_tokens(
            self.tokenizer.encode(text, return_tensors="pt").to(device)
        )

    def get_embeddings_translation(self, input_embeddings):
        return self.tokenizer.decode(
            self.model.generate(
                inputs_embeds=input_embeddings.to(device) * self.embed_scale
            )[0], skip_special_tokens=True)

    def call_with_input_embeddings(self, input_embeddings, translation_ids):
        return self.model(
            inputs_embeds=input_embeddings.to(device) * self.embed_scale,
            labels=translation_ids
        )

    def get_loss(self, text, translation_ids):
        inputs = self.tokenizer.encode(text, return_tensors="pt")
        return self.model(
            inputs.to(device), labels=translation_ids.to(device), output_hidden_states=True).loss

    def get_embedding_to_token_id(self, input_embedding):
        known_token = torch.where((self.internal_embeddings == input_embedding).all(dim=1))[0]
        if known_token.size(0) > 0:
            return known_token
        return F.cosine_similarity(self.internal_embeddings, input_embedding).argmax()

    def get_closest_internal_embedding(self, input_embedding):
        token_id = self.get_embedding_to_token_id(input_embedding)
        return self.internal_embeddings[token_id]

    def get_closest_internal_embeddings(self, input_embeddings):
        return torch.stack([self.get_closest_internal_embedding(input_embedding)
                            for input_embedding in input_embeddings[0]]).unsqueeze(0)

    def get_closest_relevant_internal_embedding(self, input_embedding):
        cos_sim = F.cosine_similarity(self.relevant_internal_embeddings, input_embedding)
        return self.relevant_internal_embeddings[cos_sim.argmax()]

    def sort_relevant_internal_embeddings(self, input_embedding):
        timer_start = timer()
        sorted_cos_sim = F.cosine_similarity(self.relevant_internal_embeddings, input_embedding.unsqueeze(0)) \
            .argsort(descending=True)
        timer_end = timer()
        print(f"Sorting took {timer_end - timer_start} seconds")
        return torch.stack([self.relevant_internal_embeddings[i] for i in sorted_cos_sim], dim=0).unsqueeze(0)

    def get_closest_sentence(self, input_embeddings):
        return self.get_embeddings_translation(
            input_embeddings.unsqueeze(0)
        )

    def get_embeddings_to_token_ids(self, input_embeddings):
        return torch.tensor([self.get_embedding_to_token_id(embedding) for embedding in input_embeddings[0]])

    def filter_embeddings(self, condition):
        return torch.stack([embedding for embedding in self.internal_embeddings if condition(embedding)])

