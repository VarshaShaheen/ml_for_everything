from transformers import BertTokenizer, BertModel
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

model_path = "./bart_large_cnn"
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)


def summarize(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs['input_ids'],
        num_beams=6,  # Increased beams for a broader search space
        max_length=300,  # Increased max length for more comprehensive summaries
        length_penalty=2.0,  # Apply a length penalty to encourage longer outputs
        early_stopping=False  # Consider disabling early stopping
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


# if __name__ == '__main__':
#     example_text = "Women empowerment refers to making women powerful to make them capable of deciding for themselves. Women have suffered a lot through the years at the hands of men. In earlier centuries, they were treated as almost non-existent. As if all the rights belonged to men even something as basic as voting. As the times evolved, women realized their power. There on began the revolution for women empowerment.As women were not allowed to make decisions for them, women empowerment came in like a breath of fresh air. It made them aware of their rights and how they must make their own place in society rather than depending on a man. It recognized the fact that things cannot simply work in someone’s favor because of their gender. However, we still have a long way to go when we talk about the reasons why we need it.Need for Women Empowerment.Almostevery country, no matter how progressive has a history of ill-treating women. In other words, women from all over the world have been rebellious to reach the status they have today. While the western countries are still making progress, third world countries like India still lack behind in Women Empowerment.In India, women empowerment is needed more than ever. India is amongst the countries which are not safe for women. There are various reasons for this. Firstly, women in India are in danger of honor killings. Their family thinks its right to take their lives if they bring shame to the reputation of their legacy.Moreover, the education and freedom scenario is very regressive here. Women are not allowed to pursue higher education, they are married off early. The men are still dominating women in some regions like it’s the woman’s duty to work for him endlessly. They do not let them go out or have freedom of any kind."
#     print(summarize(example_text))
