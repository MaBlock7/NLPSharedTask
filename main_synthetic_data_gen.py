import argparse
import asyncio
import datetime
import json
import os
import random
import re
import time

import openai

from typing import Any
from synthetic_data.utils import (
    load_attributes
)
from essentials.utils import load_env_variable
from essentials.data_functions import read_data
from essentials.config import (
    GOALS
)


# Data paths
SUBTOPICS = "synthetic_data/raw_data/attributes/subtopics/subtopics_new.json"
SAMPLE_DICT = {
    1: 0,    # 745
    2: 0,    # 758
    3: 0,    # 767
    4: 0,    # 754
    5: 0,    # 725
    6: 0,    # 648
    7: 0,    # 742
    8: 0,    # 769
    9: 0,    # 765
    10: 0,   # 732
    11: 0,   # 779
    12: 0,   # 775 
    13: 0,   # 703
    14: 0,   # 789
    15: 0,   # 731
    16: 0,   # 0
    17: 346  # 796
}
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
            return (
                self.prompt_list[start_index:end_index],
                self.uid_list[start_index:end_index]
            )
        else:
            raise StopIteration


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
                    4. the style of the paper should be '{style}' \n

                Return only the abstract.
            """


def construct_random_prompt_attributes(
    sdg_id: int,
    attr_dict: dict[Any]
) -> dict[Any]:
    """Sample random attributes to construct prompts."""

    # Randomly choose one of the main topics (scientific area)
    main_topic = random.sample(list(attr_dict['subtopics'][str(sdg_id)].keys()), 1)[0]

    # Randomly choose one of the subtopics within the chosen area
    sub_topic = random.sample(attr_dict['subtopics'][str(sdg_id)][main_topic]['Subtopics'], 1)[0]

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


def clean_str(string: str) -> str:
    """Cleans model output."""
    splits = string.split('Abstract')
    if len(splits) == 2:
        # Only use the actual abstract text
        string = splits[-1]
    string = re.sub(r"[^A-Za-z0-9(),.:!?\"\']", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def save_generated_examples(
    output_dir: str,
    timestamp: str,
    model: str,
    sdg_id: int,
    results: list[dict]):
    """Saves results to output directory."""

    prefix = f"{model}/sdg_goal_{sdg_id}/{timestamp}_synthetic_data.jsonl"
    os.makedirs(f"{output_dir}/{model}/sdg_goal_{sdg_id}", exist_ok=True)
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


def call_api_async(
    client,
    messages_list: list[list[dict[str, Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    loop: asyncio.AbstractEventLoop
) -> list[str]:
    """Asynchronous API calling function using an explicit loop."""
    print("===================================")
    print(f"call APIs, {len(messages_list)} in total, temp={temperature}.")

    response = loop.run_until_complete(
        dispatch_openai_requests(
            client,
            messages_list=messages_list,
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

    # Current timestamp
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

    # Read in raw data
    df = read_data(GOALS, format='csv').rename(columns={'goal': 'SDG', 'description': 'DESC'})

    # Create sdgID to label text mapping
    id2label = dict(zip(df['SDG'].unique(), df['DESC'].unique()))
    sdg_ids = list(id2label.keys())
    print(sdg_ids)

    # Define OpenAI client
    together_models = {
        'meta-llama/Llama-3-70b-chat-hf': 'llama-3-70b',
        'mistralai/Mixtral-8x22B-Instruct-v0.1': 'mixtral-8x22B'
    }
    if args.model_name in together_models.keys():
        client = openai.AsyncOpenAI(
            api_key=load_env_variable(variable_name='TOGETHER_API_KEY'),
            base_url="https://api.together.xyz/v1"
        )
    else:
        client = openai.AsyncOpenAI(
            api_key=load_env_variable(variable_name='OPENAI_API_KEY')
        )

    # Create attribute dictionary with choices for random sampling
    attr_dict = {attr: process_attributes(attr) for attr in args.attributes}

    # Prepare for asynchronous operations
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Produce examples for each sdg goal
    for sdg_id in sdg_ids:

        print(f"SDG Goal: {id2label[sdg_id]}.")

        # Set random seed
        random.seed(int(sdg_id + 1234))

        # Create list of random attributes dictionaries
        prompt_attributes = [construct_random_prompt_attributes(sdg_id, attr_dict) for _ in range(args.n_sample[sdg_id])]

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
                        args.top_p,
                        loop
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
                        timestamp,
                        together_models.get(args.model_name, args.model_name),
                        sdg_id,
                        final_results
                    )

                    # Stop if successful request
                    break

                # Handel error cases
                except openai.RateLimitError:
                    print("Rate Limit Error! Attempt:", attempts)
                    time.sleep(30)

                except openai.APIConnectionError:
                    print("APIConnectionError! Attempt:", attempts)
                    time.sleep(10)

                except openai.APIError as e:
                    print(f"API Error! {e} Attempt:", attempts)
                    time.sleep(10)

                finally:
                    attempts += 1

    loop.close()


if __name__ == '__main__':

    # Init parser and add arguments
    parser = argparse.ArgumentParser('')

    parser.add_argument(
        '--model_name',
        default='gpt-3.5-turbo',
        choices=['gpt-3.5-turbo', 'gpt-4-turbo', 'meta-llama/Llama-3-70b-chat-hf', 'mistralai/Mixtral-8x22B-Instruct-v0.1'],
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
        '--batch_size',
        default=20,
        type=int,
        help='number of generated examples per batch'
    )
    parser.add_argument(
        '--max_tokens',
        default=512,
        type=int,
        help='the maximum number of tokens for generation'
    )
    parser.add_argument(
        '--output_dir',
        default='synthetic_data',
        type=str,
        help='the folder for saving the generated text'
    )

    args = parser.parse_args()

    # Add specific arguments
    args.attributes = ["length", "subtopics", "style"]
    args.n_sample = SAMPLE_DICT

    main(args)
