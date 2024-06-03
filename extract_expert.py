# inspired by https://github.com/cognitivecomputations/extract-expert/blob/main/extract.py
import argparse
import json
import os
import torch
from safetensors.torch import safe_open, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

def extract_expert_model(model_name, output_dir, expert_idx):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_layers = model.config.num_hidden_layers

    # Update the configuration to reflect extraction of a single expert
    model.config.num_experts_per_tok = 1
    model.config.num_local_experts = 1

    weight_map_file = os.path.join(model_name, "model.safetensors.index.json")
    with open(weight_map_file, "r") as f:
        weight_map_data = json.load(f)
    weight_map = weight_map_data["weight_map"]

    print("Starting expert extraction...")
    extracted_weights = {}
    for layer_i in range(num_layers):
        print(f"Processing layer {layer_i}...")

        # Extract gate weights for the specified expert
        gate_weight_name = f"model.layers.{layer_i}.block_sparse_moe.gate.weight"
        gate_weight_file = os.path.join(model_name, weight_map[gate_weight_name])
        with safe_open(gate_weight_file, framework="pt") as f:
            gate_weights = f.get_tensor(gate_weight_name)
        extracted_gate_weight = gate_weights[:, expert_idx].unsqueeze(1)
        extracted_weights[gate_weight_name] = extracted_gate_weight

        # Extract weights for the specified expert
        for weight_type in ["w1", "w2", "w3"]:
            weight_name = f"model.layers.{layer_i}.block_sparse_moe.experts.{expert_idx}.{weight_type}.weight"
            weight_file = os.path.join(model_name, weight_map[weight_name])
            with safe_open(weight_file, framework="pt") as f:
                weight = f.get_tensor(weight_name)
            extracted_weights[weight_name] = weight

    # Process non-expert and non-gate weights
    for weight_name, weight_file in weight_map.items():
        if "block_sparse_moe" not in weight_name:
            with safe_open(os.path.join(model_name, weight_file), framework="pt") as f:
                weight = f.get_tensor(weight_name)
            extracted_weights[weight_name] = weight

    print("Expert extraction completed.")

    # Save the extracted weights
    extracted_weight_map = {}
    for weight_name, weight_tensor in extracted_weights.items():
        shard_file_name = "model.safetensors"
        extracted_weight_map[weight_name] = shard_file_name

    shard_file_path = os.path.join(output_dir, "model.safetensors")
    save_file(extracted_weights, shard_file_path)
    print(f"Extracted weights saved to {shard_file_path}")

    # Save the weight map to a JSON file
    weight_map_file = os.path.join(output_dir, "model.safetensors.index.json")
    with open(weight_map_file, "w") as f:
        json.dump({
            "metadata": {
                "total_size": sum(tensor.numel() * tensor.element_size() for tensor in extracted_weights.values()),
                "format": "pt",
                "pytorch_version": torch.__version__,
            },
            "weight_map": extracted_weight_map
        }, f)
    print(f"Weight map saved to {weight_map_file}")

    # Save the modified configuration and tokenizer
    model.config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Modified configuration and tokenizer saved to {output_dir}")

    # Get a list of all safetensor files in the output directory
    safetensor_files = [file for file in os.listdir(output_dir) if file.endswith(".safetensors")]

    for safetensor_file in safetensor_files:
        safetensors_path = os.path.join(output_dir, safetensor_file)
        tensors = dict()
        try:
            # Open the safetensors file in read mode
            with safe_open(safetensors_path, framework="pt") as f:
                # Iterate over all keys in the safetensors file
                for key in f.keys():
                    # Load each tensor using its key and store it in the 'tensors' dictionary
                    tensors[key] = f.get_tensor(key)
            # Save the tensors back to the safetensors file with added metadata
            save_file(tensors, safetensors_path, metadata={'format': 'pt'})
            print(f"Tensors in {safetensor_file} have been successfully saved with metadata.")
        except Exception as e:
            print(f"An error occurred for {safetensor_file}: {str(e)}")

    # Load the model from the safetensors file
    model = AutoModelForCausalLM.from_pretrained(
        output_dir,
        config=model.config,
        ignore_mismatched_sizes=True,
        torch_dtype="auto",
    )

    # Save the model, configuration, and tokenizer to the output directory
    model.save_pretrained(output_dir, max_shard_size="10GB")
    model.config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model, configuration, and tokenizer saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, help="Name or path of the model repository on the Hugging Face Hub")
    parser.add_argument("--output-dir", required=True, help="Location to write the extracted HF model")
    parser.add_argument("--expert-idx", type=int, required=True, help="Index of the expert to extract")
    args = parser.parse_args()
    extract_expert_model(
        model_name=args.model_name,
        output_dir=args.output_dir,
        expert_idx=args.expert_idx,
    )

if __name__ == "__main__":
    main()
