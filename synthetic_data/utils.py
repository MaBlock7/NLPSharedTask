import re


def remove_numbered_lists(text):
    # Use regular expression to match numbered lists
    pattern = r'^\d+\.\s+'
    regex = re.compile(pattern, re.MULTILINE)
    # Remove numbered lists from the text
    text = regex.sub('', text)
    return text


def load_attributes(attr_name: str = '') -> dict | list:
    """Returns a dictionary for class-dependent features, and a list for generic features"""
    lst = []
    with open(f"synthetic_data/raw_data/attributes/{attr_name}/{attr_name}.txt", 'r') as f:
        for lines in f:
            lst.append(lines.strip("\n"))
    return lst


def init_default_parser():

    parser = argparse.ArgumentParser('')

    parser.add_argument(
        '--model_name',
        default='gpt-3.5-turbo',
        type=str,
        help='name of the openAI model to use'
    )
    parser.add_argument(
        '--temperature',
        default=1.0,
        type=float
    )
    parser.add_argument(
        '--top_p',
        default=1.0,
        type=float
    )
    parser.add_argument(
        '--n_sample',
        default=100,
        type=int,
        help='number of generated examples per class'
    )
    parser.add_argument(
        '--batch_size',
        default=20,
        type=int,
        help='number of generated examples per batch'
    )
    parser.add_argument(
        '--max_tokens',
        default=650,
        type=int,
        help='the maximum number of tokens for generation'
    )
    parser.add_argument(
        '--output_dir',
        default='synthetic_data/produced_data',
        type=str,
        help='the folder for saving the generated text'
    )

    return parser