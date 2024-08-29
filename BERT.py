# BERT (Bidirectional Encoder Representations from Transformers)

from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained model tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Encode some text
text = "Here is some text to encode."
encoded_input = tokenizer(text, return_tensors='pt')

# Load pre-trained model
model = BertModel.from_pretrained('bert-base-uncased')

# Get hidden states
with torch.no_grad():
    outputs = model(**encoded_input)
    last_hidden_states = outputs.last_hidden_state

print(last_hidden_states)
