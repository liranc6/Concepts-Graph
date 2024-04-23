# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List, Optional
import fire
from llama import Dialog, Llama
import json
from tqdm import tqdm
from itertools import islice
import torch
import re
import os
import time 

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

def clear_file(file_path):
    with open(file_path, 'w') as f:
        pass

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialogs: List[Dialog] = [
        [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
        [
            {"role": "user", "content": "I am going to Paris, what should I see?"},
            {
                "role": "assistant",
                "content": """\
Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
            },
            {"role": "user", "content": "What is so great about #1?"},
        ],
        [
            {"role": "system", "content": "Always answer with Haiku"},
            {"role": "user", "content": "I am going to Paris, what should I see?"},
        ],
        [
            {
                "role": "system",
                "content": "Always answer with emojis",
            },
            {"role": "user", "content": "How to go from Beijing to NY?"},
        ],
    ]
    results = generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")

# Function to save subset metadata to the file
def append_to_file(subset_metadata, json_file_path):
    with open(json_file_path, 'a') as json_file:
        for e in subset_metadata:
            json.dump(e, json_file)
            json_file.write('\n')  # Add newline for readability

def str_list_to_lower(lst):
    return [x.lower() for x in lst]

def run_llama():
    ckpt_dir = "/home/liranc6/con_graph/llama3/Meta-Llama-3-8B-Instruct"
    tokenizer_path = "/home/liranc6/con_graph/llama3/Meta-Llama-3-8B-Instruct/tokenizer.model"
    temperature = 0.6
    top_p = 0.9
    max_seq_len = 800
    max_batch_size = 50
    max_gen_len = 600
    resume = True

    torch.set_default_device('cuda')

    # Create a Llama generator
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    #create a dialog
    subset_metadata_path_without_concepts = "/mnt/qnap/liranc6/data/con_graph/dataset/subset_metadata.json"
    subset_metadata_path_with_concepts = "/mnt/qnap/liranc6/data/con_graph/dataset/subset_metadata_with_concepts.json"
    corpus_path = "/home/liranc6/con_graph/corpus.txt"
    if resume:
        with open(corpus_path, 'r') as corpus_file:
            corpus = set(corpus_file.read().splitlines())
            print(len(corpus))
    else:
        corpus = set()

    if not resume:
        clear_file(subset_metadata_path_with_concepts)

    # Count the number of lines in the file
    with open(subset_metadata_path_without_concepts, 'r') as json_file:
        num_lines = sum(1 for _ in json_file)
    

    num_lines = 8000
    num_concepts_per_paper = 10

    save_for_plts_path = f"/mnt/qnap/liranc6/data/con_graph/dataset/save_for_plts_{num_concepts_per_paper}.json"
    if resume:
        with open(save_for_plts_path, 'r') as save_for_plts_file:
            #load list of dicts from json file
            save_for_plts = json.load(save_for_plts_file)
    else:
        save_for_plts = []

    with open(subset_metadata_path_without_concepts, 'r') as json_file:
        # Read the first 10 lines
        # Define a constant for the number of lines to load at a time
        LINES_PER_BATCH = 7

        # Calculate the total number of batches
        num_batches = num_lines // LINES_PER_BATCH

        to_damp = []

        dialogs: List[Dialog] = []
        print(f"Processing {num_lines} papers")

        # Initialize time tracker
        start_time = time.time()

        progress_bar = tqdm(total=num_batches, desc="Processing batches")

        with open(subset_metadata_path_with_concepts, 'r') as resume_file:
            num_lines_done = sum(1 for _ in resume_file)

        json_file = islice(json_file, num_lines_done, None)
        for i, line in enumerate(json_file, start=num_lines_done):

            paper = json.loads(line)
            dialog = [{"role": "system", "content": f"Always answer with a list size {num_concepts_per_paper}, each item is up to 3 words long"},
                    {"role": "user", "content": f"provide a list of {num_concepts_per_paper} concepts from computer science (i.e. index_terms, topics, keywords, models names) from this paper. They should give other researchers an idea of what the paper is about. Arranged from the most general to the most specific.\n" \
                                                f"paper name: {paper['title']}\n"\
                                                f"abstract: {paper['abstract']}"}]

            dialogs.append(dialog)
            to_damp.append(paper)

            if i % LINES_PER_BATCH == 0:

                batch = i // LINES_PER_BATCH

                # Update tqdm description
                results = generator.chat_completion(
                    dialogs,
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                )

                for paper, result in zip(to_damp, results):
                    concepts = result['generation']['content']
                    concepts = re.findall(r'\d+\.\s(.*)', concepts)
                    paper['concepts'] = str_list_to_lower(concepts)
                    to_damp.append(paper)
                    corpus.update(concepts)

                append_to_file(to_damp, subset_metadata_path_with_concepts)

                # Calculate average time per iteration
                if progress_bar.n > 0:
                    avg_time_per_iteration = progress_bar.format_dict['elapsed'] / progress_bar.n
                else:
                    avg_time_per_iteration = 0

                # Calculate remaining time
                remaining_time = (progress_bar.total - progress_bar.n) * avg_time_per_iteration
                #to time:
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))

                # Update tqdm description
                progress_bar.set_postfix({
                    'lines': f"{i}/{num_lines}", 
                    'batch': f"{progress_bar.n+1}/{progress_bar.total}", 
                    'remain': f"{remaining_time}",
                    'remaining': progress_bar.format_dict.get('remaining', 'unknown')
                })
                progress_bar.update(1)

                tqdm.write(f"Corpus size: {len(corpus)}/{i*num_concepts_per_paper} - avg vertex that wasnt created per paper: {(i*num_concepts_per_paper-len(corpus))/i:.2f}")
                save_for_plts.append({
                    'num_papers': i,
                    'corpus_size': len(corpus),
                    })

                # Clear lists for next batch
                to_damp.clear()
                dialogs.clear()
                
                if batch % 30 == 0:
                    with open(corpus_path, 'w') as corpus_file:
                        dump_corpus = '\n'.join(list(corpus))    
                        corpus_file.write(dump_corpus + '\n')

                    with open(save_for_plts_path, 'w') as save_for_plts_file:
                        json.dump(save_for_plts, save_for_plts_file)


        progress_bar.close()
        
        # Calculate and print total processing time
        total_time = time.time() - start_time
        print(f"Total processing time: {total_time:.2f} seconds")     
        
        with open(corpus_path, 'w') as corpus_file:
            corpus = '\n'.join(corpus)    
            corpus_file.write(corpus + '\n')

        with open(save_for_plts_path, 'w') as save_for_plts_file:
            json.dump(save_for_plts, save_for_plts_file)


if __name__ == "__main__":
    fire.Fire(run_llama)
