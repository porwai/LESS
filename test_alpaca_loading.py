import os
from transformers import AutoTokenizer
# Assuming your modified script is accessible in the PYTHONPATH
# Or adjust the path accordingly
from less.data_selection.get_validation_dataset import get_dataset, get_dataloader

# --- Configuration ---
# Path to the base model used for tokenization (should match training/warmup)
MODEL_PATH = "meta-llama/Llama-2-7b-hf"
# Path to your main data directory
DATA_DIR = "../data"
# Max sequence length used in LESS steps
MAX_LENGTH = 2048 # Or whatever you used
# Chat format used (must match training/warmup and your function)
CHAT_FORMAT = "tulu" # Or "llama-chat", etc.
# --- End Configuration ---

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
# Important: Set pad token if not already set (Llama often needs this)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded.")

try:
    # Call the main function to get your AlpacaEval dataset
    alpaca_dataset = get_dataset(
        task="alpacaeval",
        data_dir=DATA_DIR,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        use_chat_format=True, # Match your setup
        chat_format=CHAT_FORMAT # Match your setup
    )
    print(f"Dataset loaded successfully. Number of examples: {len(alpaca_dataset)}")

    # Get a dataloader (optional, but helps check collation)
    dataloader = get_dataloader(alpaca_dataset, tokenizer, batch_size=1)
    print("Dataloader created.")

    # Inspect the first few examples
    print("\n--- Inspecting first 3 examples ---")
    for i, batch in enumerate(dataloader):
        if i >= 3:
            break
        print(f"\n--- Example {i+1} ---")
        # Decode input_ids
        input_text = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False)
        print(f"** Decoded Input ({batch['input_ids'].shape}):**\n{input_text}")

        # Decode labels (masking -100)
        labels = batch['labels'][0]
        labels[labels == -100] = tokenizer.pad_token_id # Replace -100 to decode
        label_text = tokenizer.decode(labels, skip_special_tokens=False)
        print(f"** Decoded Labels ({batch['labels'].shape}):**\n{label_text}")

        # Check attention mask length
        print(f"** Attention Mask Length:** {len(batch['attention_mask'][0])}")

except FileNotFoundError as e:
    print(f"\nERROR: File not found. Did you place alpaca_eval.json correctly?")
    print(e)
except ValueError as e:
    print(f"\nERROR: Problem processing data. Check JSON format, field names, or chat format.")
    print(e)
except Exception as e:
    print(f"\nAn unexpected error occurred:")
    print(e)