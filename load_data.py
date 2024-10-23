import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import T5TokenizerFast
import torch
import nltk

nltk.download("punkt")

tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
PAD_IDX = tokenizer.pad_token_id


def load_abbreviations(path):
    """
    Load abbreviations from a file into a dictionary.
    Args:
        path (str): Path to the file containing abbreviations.
    Returns:
        dict: A dictionary mapping full forms to their abbreviations.
    """
    abbreviations = {}
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            full, abbr = line.strip().split("\t")
            abbreviations[full] = abbr
    return abbreviations


def preprocess_text(lines, abbreviations):
    """
    Replace full forms in the text lines with their abbreviations.
    Args:
        lines (list of str): The text lines to preprocess.
        abbreviations (dict): A dictionary mapping full forms to abbreviations.
    Returns:
        list of str: The preprocessed text lines.
    """
    processed_lines = []
    for line in lines:
        for full, abbr in abbreviations.items():
            # Replace full form with abbreviation only if the full form stands as a whole word
            pattern = r"\b" + re.escape(full) + r"\b"
            line = re.sub(pattern, abbr, line)
        processed_lines.append(line)
    return processed_lines


class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        """
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output.
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        """
        self.split = split
        self.input_ids = []
        self.attention_mask = []
        self.target_ids = []
        self.target_mask = []
        self.abbreviations = load_abbreviations("data/alignment.txt")
        self.process_data(data_folder, split)

    def process_data(self, data_folder, split):
        """
        Process the data by loading and tokenizing the natural language and SQL queries.

        Args:
            data_folder (str): The directory where the data files are stored.
            split (str): The dataset split - typically 'train', 'dev', or 'test'.

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]: Tuple of lists containing tokenized input ids and target ids.
        """
        input_path = os.path.join(data_folder, f"{split}.nl")
        target_path = os.path.join(data_folder, f"{split}.sql")

        inputs = self.load_and_tokenize(input_path, "input")
        if "test" in target_path:
            targets = []
        else:
            targets = self.load_and_tokenize(target_path, "target")

        return inputs, targets

    def load_and_tokenize(self, filepath, type):
        """
        Load text data from a file and tokenize it.

        Args:
            filepath (str): Path to the text file.

        Returns:
            List[torch.Tensor]: List of tokenized text data.
        """
        with open(filepath, "r", encoding="utf-8") as file:
            lines = [line.strip() for line in file.readlines()]
            # Preprocess lines to replace full forms with abbreviations
            lines = preprocess_text(lines, self.abbreviations)

        if type == "input":
            task_prefix_prompt = "Convert English natural language to a SQL query:\n"
            lines = [task_prefix_prompt + line for line in lines]
            tokenized = tokenizer(
                lines,
                max_length=256,
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True,
                add_special_tokens=True,
                padding="longest",
            )
            self.input_ids.extend(tokenized.input_ids)
            self.attention_mask.extend(tokenized.attention_mask)
        elif type == "target":
            lines = ["<extra_id_0>" + line + "</s>" for line in lines]
            tokenized = tokenizer(
                lines,
                max_length=256,
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True,
                add_special_tokens=True,
                padding="longest",
            )
            self.target_ids.extend(tokenized.input_ids)
            self.target_mask.extend(tokenized.attention_mask)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "encoder_ids": self.input_ids[idx],
            "encoder_mask": self.attention_mask[idx],
        }

        if self.split != "test":
            item.update(
                {
                    "decoder_inputs": self.target_ids[idx][:-1],
                    "decoder_targets": self.target_ids[idx][1:],
                }
            )

        return item


def normal_collate_fn(batch):
    """
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    """
    encoder_input = [item["encoder_ids"] for item in batch]
    encoder_mask = [item["encoder_mask"] for item in batch]
    decoder_input = [item["decoder_inputs"] for item in batch]
    decoder_targets = [item["decoder_targets"] for item in batch]

    # Ensure all sequences are padded to the maximum length of any sequence in the batch
    encoder_ids = pad_sequence(encoder_input, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_mask, batch_first=True, padding_value=PAD_IDX)
    decoder_input_ids = pad_sequence(
        decoder_input, batch_first=True, padding_value=PAD_IDX
    )
    # Use -100 for targets to ignore in loss calculation
    decoder_target_ids = pad_sequence(
        decoder_targets, batch_first=True, padding_value=-100
    )
    initial_decoder_input_ids = decoder_input_ids[:, 0].unsqueeze(1)
    return (
        encoder_ids,
        encoder_mask,
        decoder_input_ids,
        decoder_target_ids,
        initial_decoder_input_ids,
    )


def test_collate_fn(batch):
    """
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns:
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    """
    encoder_input = [item["encoder_ids"] for item in batch]
    encoder_mask = [item["encoder_mask"] for item in batch]

    # Padding: pad all sequences to the maximum length of any sequence in the batch
    encoder_ids = pad_sequence(encoder_input, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_mask, batch_first=True, padding_value=PAD_IDX)
    # Initialize initial_decoder_inputs with the start token ID for the decoder
    # Assuming <extra_id_0> is the start token, and its ID is obtained from the tokenizer
    start_token_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")
    initial_decoder_inputs = encoder_ids.new_full(
        (encoder_ids.size(0), 1), fill_value=start_token_id
    )
    return encoder_ids, encoder_mask, initial_decoder_inputs


def get_dataloader(batch_size, split):
    data_folder = "data"
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = lambda batch: (
        normal_collate_fn(batch) if split != "test" else test_collate_fn(batch)
    )

    dataloader = DataLoader(
        dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )
    return dataloader


def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")

    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines


def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    dev_x = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))
    test_x = load_lines(os.path.join(data_folder, "test.nl"))
    return train_x, train_y, dev_x, dev_y, test_x


def analyze_data(token_ids):
    """
    Analyze tokenized data to calculate the number of examples, mean sentence length, and vocabulary size.
    Args:
        token_ids (List[torch.Tensor]): List of tokenized text data.
    Returns:
        tuple: Number of examples, mean sentence length, vocabulary size.
    """
    total_length = sum(len(ids) for ids in token_ids)
    mean_length = total_length / len(token_ids)
    vocab_size = tokenizer.vocab_size
    return len(token_ids), mean_length, vocab_size


def extract_token_ids_from_loader(loader):
    """
    Extract token IDs from data loader.
    Args:
        loader (DataLoader): DataLoader object.
    Returns:
        List[torch.Tensor]: List of token IDs.
    """
    all_token_ids = []
    for batch in loader:
        input_ids = batch[0]  # Assuming batch[0] contains input_ids
        all_token_ids.extend(input_ids)
    return all_token_ids


def load_and_analyze_data(path, abbreviations=None, nl=True):
    with open(path, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file.readlines()]
    if abbreviations:
        lines = preprocess_text(lines, abbreviations)
        if nl:
            task_prefix_prompt = "Convert English natural language to a SQL query:\n"
            lines = [task_prefix_prompt + line for line in lines]
        else:
            lines = ["<extra_id_0>" + line + "</s>" for line in lines]
    print(lines[1])
    return analyze_data(lines)


def main():
    data_folder = "data"
    abbreviations = load_abbreviations("data/alignment.txt")
    
    # Analyze natural language data
    nl_path = os.path.join(data_folder, "train.nl")
    num_examples, mean_sentence_length, vocab_size_nl = load_and_analyze_data(nl_path, abbreviations, nl=True)
    num_examples_pre, mean_sentence_length_pre, vocab_size_nl_pre = load_and_analyze_data(nl_path, nl=True)
    
    # Analyze SQL data
    sql_path = os.path.join(data_folder, "train.sql")
    _, mean_sql_length, vocab_size_sql = load_and_analyze_data(sql_path, abbreviations, nl=False)
    _, mean_sql_length_pre, vocab_size_sql_pre = load_and_analyze_data(sql_path, nl=False)
    
    print("Natural Language - Before Preprocessing:")
    print(f"Number of Examples: {num_examples_pre}")
    print(f"Mean Sentence Length: {mean_sentence_length_pre}")
    print(f"Vocabulary Size: {vocab_size_nl_pre}")
    print("Natural Language - After Preprocessing:")
    print(f"Number of Examples: {num_examples}")
    print(f"Mean Sentence Length: {mean_sentence_length}")
    print(f"Vocabulary Size: {vocab_size_nl}")
    
    print("SQL - Before Preprocessing:")
    print(f"Mean SQL Query Length: {mean_sql_length_pre}")
    print(f"Vocabulary Size: {vocab_size_sql_pre}")
    print("SQL - After Preprocessing:")
    print(f"Mean SQL Query Length: {mean_sql_length}")
    print(f"Vocabulary Size: {vocab_size_sql}")


if __name__ == "__main__":
    main()

