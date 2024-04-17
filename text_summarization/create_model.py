import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        input_text = sample['input_text']
        target_text = sample['target_text']

        input_encoding = self.tokenizer(input_text, truncation=True, padding='max_length', max_length=512)
        target_encoding = self.tokenizer(target_text, truncation=True, padding='max_length', max_length=64)

        input_ids = torch.tensor(input_encoding['input_ids'])
        attention_mask = torch.tensor(input_encoding['attention_mask'])
        target_ids = torch.tensor(target_encoding['input_ids'])

        return input_ids, attention_mask, target_ids


# %%
input_text_path = "your_train.txt"
summary_text_path = "your_summary.txt"
# %%
sample_data = [
    {'input_text': open(input_text_path, 'r').read(), 'target_text': open(summary_text_path, 'r').read()},
]
# %%
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
dataset = CustomDataset(sample_data, tokenizer)
# %%
batch_size = 10
learning_rate = 5e-5
num_epochs = 15
# %%
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
# %%
for epoch in range(num_epochs):
    model.train()
    for batch in dataloader:
        input_ids, attention_mask, target_ids = batch
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=target_ids[:, :-1])
        logits = outputs.logits
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), target_ids[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
    scheduler.step()


# %%
def generate_summary(input_text):
    model.eval()
    inputs = tokenizer([input_text], max_length=2048, return_tensors='pt', truncation=True)
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=500, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


# %%
input_text_path = "your_target_input.txt"
with open(input_text_path, 'r') as file:
    input_text = file.read()
summary = generate_summary(input_text)
# %%
output_summary_path = "your_target_summary.txt"
with open(output_summary_path, 'w') as file:
    file.write(summary)
model.save_pretrained('./bart_summarization_model')

