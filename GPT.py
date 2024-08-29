# GPT (Generative Pre-trained Transformer)
# Use Case: Text generation, conversational AI

from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained model tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Encode a prompt text
input_text = "Once upon a time,"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Load pre-trained model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Generate text
output = model.generate(input_ids, max_length=50, num_return_sequences=1)

# Decode the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
