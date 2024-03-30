import pandas as pd

import openai
import asyncio
import os
import re
import time
import json
import random

from typing import Any
from synthetic_data.utils import load_attributes
from essentials.utils import (
    load_env_variable,
    init_default_parser
)

# GitHub paths
ABSTRACTS = "https://raw.githubusercontent.com/ZurichNLP/sdg_swisstext_2024_sharedtask/main/data/task1_train.jsonl"
GOALS = "https://raw.githubusercontent.com/datapopalliance/SDGs/master/SDG-goals.csv"

# Data paths
SUBTOPICS = "synthetic_data/raw_data/attributes/subtopics/subtopics.json"

# ----------------
# HELPER FUNCTIONS
# ----------------

def read_data(path: str, format : str = 'jsonl') -> pd.DataFrame:
    """Reads training data from github."""
    if format == 'jsonl':
        return pd.read_json(path, lines=True)
    elif format == 'csv':
        return pd.read_csv(path)
    else:
        raise TypeError


def process_attributes(attr_name: str, sdg_label_names: list[str]) -> dict:
    """Loads attributes for each sdg"""
    if 'subtopics' in attr_name:
        return json.load(open(SUBTOPICS, "r"))
    else:
        return load_attributes(attr_name=attr_name, classes=sdg_label_names)


def prepare_prompt(sdg_goal, main_topic, subtopic, length, style):

    if sdg_goal == 'no_sdg':
        first_condition = "the paper should not be related to any UN SDG goal"
    else:
        first_condition = f"the paper has to be related to the UN SDG goal of {sdg_goal};"

    return f"""
                Write an abstract of a {main_topic} paper on Web of Science, following the requirements below: \n
                    1. {first_condition} \n
                    2. the paper abstract should focus on '{subtopic}'; \n
                    3. should be in length between {length} words and {int(length) + 60} words; \n
                    4. the style of the paper should be '{style}'
            """


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),.!?\"\']", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def save_generated_examples(output_dir: str, sdg_id: int, results: list[dict], attempt: int, top_p: float):
    """Saves results to output directory"""

    prefix = f"gen_examples/sdg_goal_{sdg_id}/train_p_{top_p}_{attempt}.jsonl"
    os.makedirs(f"{output_dir}/gen_examples/sdg_goal_{sdg_id}", exist_ok=True)
    with open(f"{output_dir}/{prefix}", 'w') as f:
        for example in results:
            f.write(json.dumps(example) + "\n")

# ---------------
# ASYNC FUNCTIONS
# ---------------

async def dispatch_openai_requests(
    messages_list: list[list[dict[str, Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> list[str]:
    """Dispatches requests to OpenAI API asynchronously.

    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)


def call_api_async(msg_lst, model, temperature, max_tokens):
    print("===================================")
    print(f"call APIs, {len(msg_lst)} in total, temp={temperature}.")

    response = asyncio.run(
        dispatch_openai_requests(
            messages_list=msg_lst,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1.0,
        )
    )

    ans = [x['choices'][0]['message']['content'] for x in response]
    print(f"API returns {len(ans)} in total.")
    print("===================================")
    return ans

# -------------
# MAIN FUNCTION
# -------------

def main(args):

    # Read in raw data
    abstract_df = read_data(ABSTRACTS) 
    goals_df = read_data(GOALS, format='csv').rename(columns={'goal': 'SDG', 'description': 'DESC'})

    # Merge DataFrames to working data
    df = abstract_df.merge(goals_df, on='SDG', how='left')
    df.loc[df.DESC.isna(), 'DESC'] = 'no_sdg'

    id2label = dict(zip(df['SDG'].unique(), df['DESC'].unique()))
    # id2label = {v: k for k, v in label2id.items()}

    sdg_ids = list(id2label.keys())
    print(sdg_ids)

    model = args.model_name
    openai.api_key = args.api_key

    attr_dict = {attr: process_attributes(attr, sdg_id) for attr in args.attributes}

    # Produce examples for each sdg goal
    for sdg_id in sorted(sdg_ids):

        print(f"SDG Goal: {id2label[sdg_id]}.")

        sent_cnt = 0
        attempt = 0
        prompt_list = []  # To send prompts in batches
        prompt_attributes = []  # To store attributes of the prompt used
        results = []  # To store returned messages

        random.seed(sdg_id + 1234)

        while sent_cnt < args.n_sample:

            # Randomly choose one of the main topics (scientific area)
            main_topic = random.sample(list(attr_dict['subtopics'].keys()), 1)[0]

            # Randomly choose one of the subtopics within the chosen area
            sub_topic = random.sample(attr_dict['subtopics'][main_topic], 1)[0]

            style = random.sample(attr_dict["style"], 1)[0]
            length = random.sample(attr_dict["length"], 1)[0]

            attributes_dict = {'sdg_id': sdg_id,
                               'main_topic': main_topic,
                               'sub_topic': sub_topic,
                               'style': style,
                               'length': length}
            prompt_attributes.append(attributes_dict)

            # Create attribute-specific prompt and append it to the list
            prompt = prepare_prompt(id2label[sdg_id],
                                    main_topic,
                                    sub_topic,
                                    length,
                                    style)

            prompt_list.append([{'role': 'user', 'content': prompt}])

            if len(prompt_list) == args.batch_size:
                try:
                    attempt += 1
                    return_msg = call_api_async(prompt_list, model, args.temperature, args.max_tokens)

                    assert len(return_msg) == len(prompt_attributes)

                    valid = 0
                    examples = []
                    for (msg, attr) in zip(return_msg, prompt_attributes):
                        if any(word in msg for word in ['I apologize', 'sorry', 'an AI language model']):  # invalid contents
                            continue
                        else:
                            valid += 1
                            example = {'text': clean_str(msg)}
                            example.update(attr)
                            results.append(example)

                    sent_cnt += valid 
                    prompt_list = []
                    prompt_attributes = []

                    save_generated_examples(args.output_dir, sdg_id, examples, attempt, args.top_p)

                # Handel error cases
                except openai.error.RateLimitError:
                    print("Rate Limit Error! Attempt:", attempt)
                    prompt_list = []
                    prompt_attributes = []
                    time.sleep(10)
                    continue

                except  openai.error.APIError:
                    print("API Error! Attempt:", attempt)
                    prompt_list = []
                    prompt_attributes = []
                    time.sleep(5)
                    continue

                except openai.error.APIConnectionError:
                    print("APIConnectionError", attempt)
                    prompt_list = []
                    prompt_attributes = []
                    time.sleep(5)
                    continue 

                except openai.error.InvalidRequestError:
                    print("API Error! Invalid Request:", attempt)
                    prompt_list = []
                    prompt_attributes = []
                    continue

            if sent_cnt > args.n_sample or attempt >= 5:
                break


if __name__ == '__main__':

    # Init parser and add arguments
    parser = init_default_parser()
    args = parser.parse_args()

    # Add specific arguments
    args.api_key = load_env_variable(variable_name='OPENAI_API_KEY')
    args.domain = 'scientific paper'
    args.attributes = ["length", "subtopics", "style"]
    args.metadata = ""

    main(args)
