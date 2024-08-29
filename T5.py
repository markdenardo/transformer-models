# T5 (Text-To-Text Transfer Transformer)
# Use Case: Text summarization, translation, etc.

from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load pre-trained model tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Encode a text
input_text = "summarize: The quick brown fox jumps over the lazy dog."
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Load pre-trained model
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Generate summary
outputs = model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)

# Decode the generated text
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(summary)
