import os, argparse, random
from tqdm import tqdm
import numpy as np
import json
import torch
from transformers import GemmaTokenizerFast, GemmaForCausalLM
from transformers import GemmaTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

from utils import (
    set_random_seeds,
    compute_metrics,
    save_queries_and_records,
    compute_records,
)
from prompting_utils import read_schema, extract_sql_query, save_logs
from load_data import load_prompting_data

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)  # you can add mps
MAX_NEW_TOKENS = 512


def get_args():
    """
    Arguments for prompting. You may choose to change or extend these as you see fit.
    """
    parser = argparse.ArgumentParser(
        description="Text-to-SQL experiments with prompting."
    )

    parser.add_argument(
        "-s",
        "--shot",
        type=int,
        default=3,
        help="Number of examples for k-shot learning (0 for zero-shot)",
    )
    parser.add_argument("-p", "--ptype", type=int, default=0, help="Prompt type")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gemma",
        help="Model to use for prompting: gemma (gemma-1.1-2b-it) or codegemma (codegemma-7b-it)",
    )
    parser.add_argument(
        "-q",
        "--quantization",
        action="store_true",
        help="Use a quantized version of the model (e.g. 4bits)",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed to help reproducibility"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="experiment",
        help="How should we name this experiment?",
    )
    args = parser.parse_args()
    return args


def create_prompt(sentence, k, nl_freq_map):
    """
    Function for creating a prompt for zero or few-shot prompting.

    Add/modify the arguments as needed.

    Inputs:
        * sentence (str): A text string
        * k (int): Number of examples in k-shot prompting
    """
    prompt = "TASK:\nYou are an SQL expert. Your task is to convert English natural language queries into SQL. The natural language query will be indicated by 'NL:', and your SQL response should follow 'SQL:'.\n"    
    schema = read_schema("data/flight_database.schema")
    prompt = parse_schema(prompt, schema)
    prompt = add_kshots_to_prompt(prompt, sentence, k, nl_freq_map, random_sample=False)
    prompt += "NOTES:\nStart your response with 'SQL:'. End your response with 'YOUR TURN'.\n\n"
    prompt += f"YOUR TURN:\nNL: {sentence}\nSQL:"
    return prompt


def parse_schema(prompt, schema):
    ents = schema['ents']
    links = schema['links']
    prompt += "DATABASE CONTEXT:\n"
    prompt += "Schema Details:\n"
    prompt += "Entity Attributes:\n"
    for entity_category, details in ents.items():
        prompt += f"\n{entity_category} includes attributes: "
        for entity in details.keys():
            prompt += f"{entity}, "
        prompt = prompt.rstrip(', ')
    prompt += "\n\n"
    prompt += "Entity Relationships:\n"
    for source_entity, relations in links.items():
        if relations:
            for target_entity, link_field in relations.items():
                prompt += f"{source_entity} is linked to {target_entity} through {link_field}\n"
        else:
            prompt += f"{source_entity} is not linked to any other entities.\n"
    return prompt


def add_kshots_to_prompt(prompt, sentence, k, nl_freq_map, random_sample):
    """
    Function that takes in the sentence and returns k-shot examples that are most relevant to the sentence
    """
    if k <= 0:
        return prompt

    prompt += "\nEXAMPLES:\n"
    with open("data/train.sql", "r", encoding="utf-8") as file:
        sql_queries = file.readlines()
    with open("data/train.nl", "r", encoding="utf-8") as file:
        nl_queries = file.readlines()

    if random_sample:
        random_indices = random.sample(range(len(nl_queries)), k)
        for index in random_indices:
            prompt += f"NL: {nl_queries[index]}\nSQL: {sql_queries[index]}\n"
    else:
        sentence_freq_map = nl_to_freq(sentence)
        # Calculate similarities and get the indices of the k most similar entries
        similarities = [
            cosine_similarity(sentence_freq_map, freq_map) for freq_map in nl_freq_map
        ]
        # Get indices of k largest values
        most_relevant_indices = np.argsort(similarities)[-k:]
        for i, index in enumerate(most_relevant_indices):
            prompt += f"Ex {i+1}:\nNL: {nl_queries[index]}\nSQL: {sql_queries[index]}\n"
    return prompt


def initialize_nl_freq_map():
    with open("data/train.nl", "r", encoding="utf-8") as file:
        return [nl_to_freq(nl_query) for nl_query in file.readlines()]


def nl_to_freq(nl_query):
    words = nl_query.strip().split()
    freq_map = {}
    for word in words:
        if word in freq_map:
            freq_map[word] += 1
        else:
            freq_map[word] = 1
    return freq_map


def cosine_similarity(freq_map1, freq_map2):
    """
    Compute cosine similarity between two frequency maps.
    """
    # Convert frequency maps to vectors
    words = set(freq_map1.keys()).union(set(freq_map2.keys()))
    vec1 = [freq_map1.get(word, 0) for word in words]
    vec2 = [freq_map2.get(word, 0) for word in words]

    # Compute cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0  # Avoid division by zero
    else:
        return dot_product / (norm1 * norm2)


def exp_kshot(tokenizer, model, inputs, k, nl_freq_map):
    """
    k-shot prompting experiments using the provided model and tokenizer.
    This function generates SQL queries from text prompts and evaluates their accuracy.

    Add/modify the arguments and code as needed.

    Inputs:
        * tokenizer
        * model
        * inputs (List[str]): A list of text strings
        * k (int): Number of examples in k-shot prompting
    """
    raw_outputs = []
    extracted_queries = []

    for i, sentence in tqdm(enumerate(inputs)):
        extracted_query = ""
        while len(extracted_query) < 5:
            prompt = create_prompt(sentence, k, nl_freq_map)  # Looking at the prompt may also help

            input_ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            outputs = model.generate(
                **input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.7
            )  # You should set MAX_NEW_TOKENS
            response = tokenizer.decode(
                outputs[0]
            )  # How does the response look like? You may need to parse it
            raw_outputs.append(response)

            # Extract the SQL query
            extracted_query = extract_sql_query(response)
            extracted_queries.append(extracted_query)
            print("Response:", response)
            print("Extracted:", extracted_query)
            if len(extracted_query) < 5:
                print("Query too short, RERUNNING!:", extracted_query)
    return raw_outputs, extracted_queries


def eval_outputs(
    eval_x, eval_y, gt_sql_path, model_sql_path, gt_record_path, model_record_path
):
    """
    Evaluate the outputs of the model by computing the metrics.

    Add/modify the arguments and code as needed.
    """
    # Compute metrics
    sql_em, record_em, record_f1, model_error_msgs = compute_metrics(
        gt_sql_path, model_sql_path, gt_record_path, model_record_path
    )

    # Calculate error rate
    num_errors = len([msg for msg in model_error_msgs if msg])
    total_queries = len(eval_x)
    error_rate = num_errors / total_queries if total_queries > 0 else 0

    return sql_em, record_em, record_f1, model_error_msgs, error_rate


def initialize_model_and_tokenizer(model_name, to_quantize=False):
    """
    Args:
        * model_name (str): Model name ("gemma" or "codegemma").
        * to_quantize (bool): Use a quantized version of the model (e.g. 4bits)

    To access to the model on HuggingFace, you need to log in and review the
    conditions and access the model's content.
    """
    if model_name == "gemma":
        model_id = "google/gemma-1.1-2b-it"
        tokenizer = GemmaTokenizerFast.from_pretrained(model_id)
        # Native weights exported in bfloat16 precision, but you can use a different precision if needed
        model = GemmaForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        ).to(DEVICE)
    elif model_name == "codegemma":
        model_id = "google/codegemma-7b-it"
        tokenizer = GemmaTokenizer.from_pretrained(model_id)
        if to_quantize:
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",  # 4-bit quantization
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, config=nf4_config
            ).to(DEVICE)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16
            ).to(DEVICE)
    return tokenizer, model


def main():
    """
    Note: this code serves as a basic template for the prompting task. You can but
    are not required to use this pipeline.
    You can design your own pipeline, and you can also modify the code below.
    """
    args = get_args()
    k = args.shot
    ptype = args.ptype
    model_name = args.model
    to_quantize = args.quantization
    experiment_name = args.experiment_name

    set_random_seeds(args.seed)

    data_folder = "data"
    train_x, train_y, dev_x, dev_y, test_x = load_prompting_data(data_folder)
    
    nl_freq_map = initialize_nl_freq_map()

    # Model and tokenizer
    tokenizer, model = initialize_model_and_tokenizer(model_name, to_quantize)

    for eval_split in ["dev", "test"]:
        eval_x, eval_y = (dev_x, dev_y) if eval_split == "dev" else (test_x, None)

        raw_outputs, extracted_queries = exp_kshot(tokenizer, model, eval_x, k, nl_freq_map)

        # # You can add any post-processing if needed
        # # You can compute the records with `compute_records``

        # # Save results
        # # You can for instance use the `save_queries_and_records` function
        model_sql_path = os.path.join(f"results/gemma_{experiment_name}_{eval_split}.sql")
        model_record_path = os.path.join(f"records/gemma_{experiment_name}_{eval_split}.pkl")
        save_queries_and_records(extracted_queries, model_sql_path, model_record_path)

        if eval_split == "dev":
            gt_query_records = "records/ground_truth_dev.pkl"
            gt_sql_path = os.path.join(f"data/{eval_split}.sql")
            sql_em, record_em, record_f1, model_error_msgs, error_rate = eval_outputs(
                eval_x,
                eval_y,
                gt_sql_path=gt_sql_path,
                model_sql_path=model_sql_path,
                gt_record_path=gt_query_records,
                model_record_path=model_record_path,
            )
            print(f"{eval_split} set results: ")
            print(f"Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
            print(
                f"{eval_split} set results: {error_rate*100:.2f}% of the generated outputs led to SQL errors"
            )

            # Save logs, if needed
            log_path = f"logs/{experiment_name}_{eval_split}.txt"
            save_logs(log_path, sql_em, record_em, record_f1, model_error_msgs)


if __name__ == "__main__":
    main()
