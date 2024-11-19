import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Experiment parser')

    # Model name
    parser.add_argument('--model_name', type=str, choices=['marian', 'deepL', 'mbart'], default='marian')

    # Source and target languages
    parser.add_argument('--src', type=str, choices=['en'], default='en')
    parser.add_argument('--trg', type=str, choices=['fr', 'de'], default='fr')

    # Sentences range
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=3000)

    # Attack type
    parser.add_argument('--attack_type', type=str, choices=['target_length', 'free'], default='free')

    # Hyperparameters
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--neighbours', type=int, default=20)
    parser.add_argument('--target_sentence', type=str, default='.')
    parser.add_argument('--input_sentence', type=str, default='.')

    return {
        arg: val for arg, val in vars(parser.parse_args()).items()
    }
