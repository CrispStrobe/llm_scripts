import argparse
import json
import math
import os
from pathlib import Path
import shutil
import requests

import torch
from tqdm import tqdm
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.sentencepiece import SentencePieceTokenizer
from safetensors.torch import save_file, load_file

FIRST_PIECE_ID = 3
OLD_VOCAB_SIZE = 32000
NEW_VOCAB_SIZE = 32768
DEFAULT_REPO = "unsloth/mistral-7b-instruct-v0.3"

class ModelArgs:
    def __init__(self, config):
        self.config = config
        self.vocab_size = config["vocab_size"]

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            config = json.load(f)
        return cls(config)

    def to_dict(self):
        return self.config

def load_sharded_model(model_path):
    model_files = sorted(model_path.glob("model-*.safetensors"))
    if not model_files:
        raise FileNotFoundError(f"No sharded safetensors files found in {model_path}")
    
    model_state_dict = {}
    print(f"Loading sharded model from {len(model_files)} files...")
    for model_file in tqdm(model_files, desc="Loading shards"):
        shard = load_file(model_file)
        for k, v in shard.items():
            if k not in model_state_dict:
                model_state_dict[k] = v
            else:
                model_state_dict[k] = torch.cat((model_state_dict[k], v), dim=0)
    
    return model_state_dict

def download_file(url, output_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def save_tokenizer(new_tokenizer_repo, extended_model, original_tokenizer_config_path):
    tokenizer_files = ["special_tokens_map.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "config.json"]
    base_url = f"https://huggingface.co/{new_tokenizer_repo}/resolve/main"

    for file_name in tokenizer_files:
        url = f"{base_url}/{file_name}"
        output_path = extended_model / file_name
        print(f"Downloading {file_name} from {url}")
        download_file(url, output_path)
        print(f"Downloaded {file_name} to {output_path}")

    # Update tokenizer_config.json with chat_template
    with open(extended_model / "tokenizer_config.json", "r") as f:
        new_tokenizer_config = json.load(f)

    with open(original_tokenizer_config_path, "r") as f:
        original_tokenizer_config = json.load(f)
    
    if "chat_template" in original_tokenizer_config:
        new_tokenizer_config["chat_template"] = original_tokenizer_config["chat_template"]

    with open(extended_model / "tokenizer_config.json", "w") as f:
        json.dump(new_tokenizer_config, f, indent=4)
    
    print("Updated tokenizer_config.json with chat_template.")

def extend_model(original_model: Path, extended_model: Path, new_tokenizer_repo: str):
    print("Starting to extend the model...")
    original_ckpt = load_sharded_model(original_model)
    
    config_path = original_model / "config.json"
    
    if not config_path.is_file():
        raise FileNotFoundError(f"No such file: '{config_path}'. Please ensure the config.json file exists in the specified directory.")

    model_args = ModelArgs.load(str(config_path))

    original_vocab_size = model_args.vocab_size
    assert (
        original_vocab_size == OLD_VOCAB_SIZE
    ), f"Original vocab size {original_vocab_size} is not equal to 32000. Can only extend models with vocab_size of 32000"

    if not extended_model.exists():
        os.makedirs(extended_model, exist_ok=True)
        print(f"Created empty directory {extended_model}.")

    assert not list(
        extended_model.iterdir()
    ), f"Make sure {extended_model} is empty"

    # Load and check tokenizers from new path
    mistral_tokenizer = MistralTokenizer.v3()
    tokenizer: SentencePieceTokenizer = mistral_tokenizer.instruct_tokenizer.tokenizer

    new_vocab_size = tokenizer.n_words
    assert (
        new_vocab_size == 32768
    ), f"New Tokenizer has vocab_size: {new_vocab_size} but has to be equal to 32768. Make sure to pass a v2 or v3 tokenizer file"

    vocabulary_delta = new_vocab_size - original_vocab_size

    # Check that 0...FIRST_PIECE_ID-1 are UNK + control characters and FIRST_PIECE_ID is the first piece
    assert tokenizer._model.id_to_piece(vocabulary_delta + FIRST_PIECE_ID) == "<0x00>"
    assert tokenizer._model.id_to_piece(FIRST_PIECE_ID - 1) == "</s>"

    assert isinstance(tokenizer, SentencePieceTokenizer)

    # Key names
    original_embeddings_key = "model.embed_tokens.weight"
    original_output_key = "lm_head.weight"

    original_embeddings = original_ckpt[original_embeddings_key]

    assert (
        original_vocab_size == original_embeddings.shape[0]
    ), f"Original vocab size {original_vocab_size} is not equal to original embeddings shape {original_embeddings.shape[0]}."

    dim = original_embeddings.shape[1]

    # Extend embeddings
    print("Extending embeddings...")
    extended_embeddings = torch.zeros(
        tokenizer.n_words, dim, dtype=original_embeddings.dtype
    )
    extended_embeddings[:original_vocab_size] = original_embeddings
    extended_embeddings[:FIRST_PIECE_ID] = original_embeddings[:FIRST_PIECE_ID]
    extended_embeddings[FIRST_PIECE_ID + vocabulary_delta :] = original_embeddings[
        FIRST_PIECE_ID:
    ]

    # randomly initialize new tokens
    extended_tokens = torch.empty(
        vocabulary_delta, dim, dtype=original_embeddings.dtype
    )
    torch.nn.init.normal_(extended_tokens, std=1 / math.sqrt(dim))

    extended_embeddings[FIRST_PIECE_ID : FIRST_PIECE_ID + vocabulary_delta] = (
        extended_tokens
    )

    # Extend output
    print("Extending output...")
    original_output = original_ckpt[original_output_key]
    assert (
        original_output.shape[0] == original_vocab_size
    ), f"Original output shape {original_output.shape[0]} is not equal to {original_vocab_size}."
    assert (
        original_output.shape[1] == dim
    ), f"Original output dim {original_output.shape[1]} is not equal to embedding dim {dim}."

    assert (
        original_output.dtype == original_embeddings.dtype
    ), f"Original output and embeddings have different dtypes: {original_output.dtype} vs {original_embeddings.dtype}."

    extended_output = torch.zeros(tokenizer.n_words, dim, dtype=original_output.dtype)
    extended_output[:FIRST_PIECE_ID] = original_output[:FIRST_PIECE_ID]
    extended_output[FIRST_PIECE_ID + vocabulary_delta :] = original_output[
        FIRST_PIECE_ID:
    ]

    # randomly initialize new tokens
    extended_tokens = torch.empty(vocabulary_delta, dim, dtype=original_output.dtype)
    torch.nn.init.normal_(extended_tokens, std=1 / math.sqrt(dim))

    extended_output[FIRST_PIECE_ID : FIRST_PIECE_ID + vocabulary_delta] = (
        extended_tokens
    )

    original_ckpt[original_embeddings_key] = extended_embeddings
    original_ckpt[original_output_key] = extended_output

    new_ckpt_path = extended_model / "consolidated.safetensors"
    print(f"Exporting extended model to {extended_model} ...")
    save_file(original_ckpt, str(new_ckpt_path))

    # Save the new config.json
    print("Saving new config.json...")
    download_file(f"https://huggingface.co/{new_tokenizer_repo}/resolve/main/config.json", extended_model / "config.json")

    # Save the new tokenizer files
    print("Saving new tokenizer files...")
    save_tokenizer(new_tokenizer_repo, extended_model, original_model / "tokenizer_config.json")

    print("Model extension and tokenizer update completed successfully.")

def main():
    parser = argparse.ArgumentParser(
        description="Extend a model using the specified original model, extended model, and tokenizer paths."
    )
    parser.add_argument(
        "--original_model_ckpt", type=Path, help="Path to the original model folder."
    )
    parser.add_argument(
        "--extended_model_ckpt", type=Path, help="Path to the extended model folder."
    )
    parser.add_argument(
        "--new_tokenizer_repo", type=str, default=DEFAULT_REPO, help="Hugging Face repository for the new tokenizer."
    )
    args = parser.parse_args()

    extend_model(
        original_model=args.original_model_ckpt,
        extended_model=args.extended_model_ckpt,
        new_tokenizer_repo=args.new_tokenizer_repo,
    )


if __name__ == "__main__":
    main()
