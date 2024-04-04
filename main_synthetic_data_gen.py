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

class BatchIterator:
    def __init__(self, prompt_list, uid_list, batch_size):
        self.prompt_list = prompt_list
        self.uid_list = uid_list
        self.batch_size = batch_size
        self.current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index < len(self.prompt_list):
            start_index = self.current_index
            self.current_index += self.batch_size
            end_index = self.current_index
            return self.prompt_list[start_index:end_index], self.uid_list[start_index:end_index]
        else:
            raise StopIteration


def read_data(path: str, format : str = 'jsonl') -> pd.DataFrame:
    """Reads training data from github."""
    if format == 'jsonl':
        return pd.read_json(path, lines=True)
    elif format == 'csv':
        return pd.read_csv(path)
    else:
        raise TypeError


def process_attributes(attr_name: str) -> dict:
    """Loads attributes for each sdg"""
    if 'subtopics' in attr_name:
        return json.load(open(SUBTOPICS, "r"))
    else:
        return load_attributes(attr_name=attr_name)


def prepare_prompt(
    sdg_goal: str,
    main_topic: str,
    subtopic: str,
    length: int,
    style: str
) -> str:
    """Fill in the prompt template with random attributes."""

    if sdg_goal == 'no_sdg':
        first_condition = "the paper should not be related to any UN SDG goal"
    else:
        first_condition = f"the paper should be related to the UN SDG goal of {sdg_goal} but do not mention the SDG goal explicitly;"

    return f"""
                Write an abstract of a {main_topic} paper on Web of Science, following the requirements below: \n
                    1. {first_condition} \n
                    2. the paper abstract should focus on '{subtopic}'; \n
                    3. should be in length between {length} words and {int(length) + 60} words; \n
                    4. the style of the paper should be '{style}'
            """


def construct_random_prompt_attributes(
    sdg_id: int,
    attr_dict: dict[Any]
) -> dict[Any]:
    """Sample random attributes to construct prompts."""

    # Randomly choose one of the main topics (scientific area)
    main_topic = random.sample(list(attr_dict['subtopics'].keys()), 1)[0]

    # Randomly choose one of the subtopics within the chosen area
    sub_topic = random.sample(attr_dict['subtopics'][main_topic], 1)[0]

    style = random.sample(attr_dict["style"], 1)[0]
    length = random.sample(attr_dict["length"], 1)[0]

    return {
        'sdg_id': int(sdg_id),
        'main_topic': main_topic,
        'sub_topic': sub_topic,
        'style': style,
        'length': int(length)
    }


def construct_prompts_from_attributes(
    sdg_description: str,
    random_attributes: dict[Any]
) -> dict[Any]:
    """Create attribute-specific prompt."""

    return prepare_prompt(
        sdg_description,
        random_attributes['main_topic'],
        random_attributes['sub_topic'],
        random_attributes['length'],
        random_attributes['style']
    )


def clean_str(string):
    """Cleans model output."""
    string = re.sub(r"[^A-Za-z0-9(),.!?\"\']", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def save_generated_examples(output_dir: str, sdg_id: int, results: list[dict], attempt: int, top_p: float):
    """Saves results to output directory."""

    prefix = f"gen_results/sdg_goal_{sdg_id}/train_p_{top_p}_{attempt}.jsonl"
    os.makedirs(f"{output_dir}/gen_results/sdg_goal_{sdg_id}", exist_ok=True)
    with open(f"{output_dir}/{prefix}", 'a') as f:  # append mode
        for example in results:
            f.write(json.dumps(example) + "\n")

# ---------------
# ASYNC FUNCTIONS
# ---------------

async def dispatch_openai_requests(
    client,
    messages_list: list[list[dict[str, Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> list[str]:
    """
    Dispatches requests to OpenAI API asynchronously.

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
        client.chat.completions.create(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)


def call_api_async(client, msg_lst, model, temperature, max_tokens, top_p):
    print("===================================")
    print(f"call APIs, {len(msg_lst)} in total, temp={temperature}.")

    response = asyncio.run(
        dispatch_openai_requests(
            client,
            messages_list=msg_lst,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
    )
    ans = [x.choices[0].message.content for x in response]

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

    # Create sdgID to label text mapping
    id2label = dict(zip(df['SDG'].unique(), df['DESC'].unique()))
    sdg_ids = list(id2label.keys())
    print(sdg_ids)

    # Define OpenAI client
    client = openai.AsyncOpenAI(
        api_key=load_env_variable(variable_name='OPENAI_API_KEY')
    )

    # Create attribute dictionary with choices for random sampling
    attr_dict = {attr: process_attributes(attr) for attr in args.attributes}

    # Produce examples for each sdg goal
    for sdg_id in sdg_ids:

        print(f"SDG Goal: {id2label[sdg_id]}.")

        # Set random seed
        random.seed(int(sdg_id + 1234))

        # Create list of random attributes dictionaries
        prompt_attributes = [construct_random_prompt_attributes(sdg_id, attr_dict) for _ in range(args.n_sample)]

        # Create list of prompts based on the previous attributes dictionaries
        prompt_list = [[{'role': 'user', 'content': construct_prompts_from_attributes(id2label[sdg_id], attributes)}] for attributes in prompt_attributes]

        # Create iterator object
        batch_iterator = BatchIterator(prompt_list, prompt_attributes, args.batch_size)

        for prompt_batch, attributes_batch in batch_iterator:

            attempts = 0

            while attempts < 3:

                try:

                    # Send async requests to the OpenAI api
                    return_msg = call_api_async(
                        client,
                        prompt_batch,
                        args.model_name,
                        args.temperature,
                        args.max_tokens,
                        args.top_p
                    )

                    # Check if for every uid a model output was returned
                    assert len(attributes_batch) == len(return_msg)

                    # Filter out unwanted outputs
                    final_results = []
                    for (msg, attr) in zip(return_msg, attributes_batch):
                        if any(word in msg for word in ['I apologize', 'sorry', 'an AI language model']):  # invalid contents
                            continue
                        else:
                            abstract = {'text': clean_str(msg)}
                            abstract.update(attr)
                            final_results.append(abstract)

                    save_generated_examples(
                        args.output_dir,
                        sdg_id,
                        final_results,
                        attempts,
                        args.top_p
                    )

                    # Stop if successful request
                    break

                # Handel error cases
                except openai.RateLimitError:
                    print("Rate Limit Error! Attempt:", attempts)
                    time.sleep(10)

                except openai.APIConnectionError:
                    print("APIConnectionError! Attempt:", attempts)
                    time.sleep(5)

                except openai.APIError as e:
                    print(f"API Error! {e} Attempt:", attempts)
                    time.sleep(5)

                finally:
                    attempts += 1


if __name__ == '__main__':

    # Init parser and add arguments
    parser = init_default_parser()
    args = parser.parse_args()

    # Add specific arguments
    args.attributes = ["length", "subtopics", "style"]

    main(args)
