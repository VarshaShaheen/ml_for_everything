from transformers import BartTokenizer, BartForConditionalGeneration
import whisper

tokenizer = BartTokenizer.from_pretrained("./bart_large_cnn")
model = BartForConditionalGeneration.from_pretrained("./bart_large_cnn")
model_whisper = whisper.load_model("base")


# Function to summarize text
def summarize(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=200, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def audio_to_text(audio_path):
    # Process the audio file
    result = model_whisper.transcribe(audio_path)
    return result['text']

