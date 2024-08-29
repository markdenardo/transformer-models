# DistilBERT (A smaller, faster version of BERT)
# Use Case: Similar to BERT but optimized for performance

from transformers import DistilBertTokenizer, DistilBertModel

# Load pre-trained model tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Encode a text
input_text = "Example text for DistilBERT."
encoded_input = tokenizer(input_text, return_tensors='pt')

# Load pre-trained model
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Get hidden states
with torch.no_grad():
    outputs = model(**encoded_input)
    hidden_states = outputs.last_hidden_state

print(hidden_states)
