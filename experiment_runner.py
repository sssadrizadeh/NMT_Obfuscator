import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MBart50TokenizerFast

from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"

# Parse the arguments and run the experiment
args = parse_args()

# Define the model
if args['model_name'] == 'marian':
    model = AutoModelForSeq2SeqLM.from_pretrained(f"Helsinki-NLP/opus-mt-{args['src']}-{args['trg']}").to(device)
    tokenizer = AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{args['src']}-{args['trg']}")
if args['model_name'] == 'mbart':
    model = AutoModelForSeq2SeqLM.from_pretrained('facebook/mbart-large-50-one-to-many-mmt').to(device)
    tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50-one-to-many-mmt')
    if args['src'] == 'en':
        tokenizer.src_lang = 'en_XX'
    if args['trg'] == 'fr':
        tokenizer.tgt_lang = 'fr_XX'
        model.config.forced_bos_token_id = tokenizer.lang_code_to_id["fr_XX"]
        
# Define the dataset and range
target = args['target_sentence']
if args['attack_type'] == 'free':
    input_sentence = args['input_sentence']
else:
    start, end = args['start'], args['end']
    dataset = [elem["en"] for elem in load_dataset("wmt14", "fr-en", split="test")["translation"]][start:end]
    


# Define the attack
if args['attack_type'] == 'target_length':
    run_length_experiment(model, tokenizer, dataset)
elif args['attack_type'] == 'free':
    run_with_target_sentence(model, tokenizer, input_sentence, target, args['iterations'])