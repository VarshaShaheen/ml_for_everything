from transformers import BartTokenizer, BartForConditionalGeneration

# Define the model name
model_name = "facebook/bart-large-cnn"  # This version is optimized for summarization

# Load the tokenizer and model
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Save both the tokenizer and model locally
model_path = "./bart_large_cnn"  # Specify your path here
tokenizer.save_pretrained(model_path)
model.save_pretrained(model_path)
