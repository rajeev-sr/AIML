import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load tokenizer & model
model_name = 'bert-base-uncased'
print(f"Loading tokenizer and model for {model_name}...")
tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
model = BertModel.from_pretrained(model_name)
print("BERT loaded successfully!")

# Put model in evaluation mode
model.eval()

# Function to get BERT embeddings for a sentence
def get_sentence_embedding(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    tokens_tensor = torch.tensor([token_ids])

    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor)
        # Take the embeddings from the last layer, [CLS] token represents sentence
        sentence_embedding = encoded_layers[-1][0][0].numpy()
    return sentence_embedding

# Example sentences
sentences = [
    "I love machine learning.",
    "Artificial intelligence is my passion.",
    "The sun is shining brightly today."
]

# Get embeddings
embeddings = [get_sentence_embedding(s) for s in sentences]

# Compute cosine similarity
similarity_matrix = cosine_similarity(embeddings)

# Display results
print("\nCosine Similarity Matrix:")
for i, s1 in enumerate(sentences):
    for j, s2 in enumerate(sentences):
        print(f"Similarity('{s1}' , '{s2}') = {similarity_matrix[i][j]:.4f}")

