import torch
from datasets import Dataset
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
import faiss
import numpy as np
from data_process import data_preprocessing


tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")

file_path = './crawling_data/scraped_data_1.xlsx.csv'
dataset_dict = data_preprocessing(file_path=file_path)

# Convert to HuggingFace dataset
dataset = Dataset.from_dict(dataset_dict)

# Tokenize the texts and add tokenized inputs to the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding='max_length', max_length=100)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type='numpy', columns=['input_ids'])

# Convert texts to embeddings
embeddings = np.array(tokenized_dataset["input_ids"])

# Check and print the dimensions of the embeddings
print("Embeddings dimensions:", embeddings.shape)

# Create and save the FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "custom_index.faiss")

# Save dataset and embeddings
dataset_path = "custom_dataset"
tokenized_dataset.save_to_disk(dataset_path)
index_path = "custom_index.faiss"

dataset.save_to_disk(dataset_path)
dataset.get_index('embeddings').save(index_path)
# Initialize the retriever with the custom index and dataset
retriever = RagRetriever.from_pretrained(
    "facebook/rag-sequence-nq",
    index_name="custom",
    passages_path=None,
    dataset_path=dataset_path,
    index_path=index_path,
    use_dummy_dataset=False
)
retriever.index = faiss.read_index(index_path)

# Initialize the model
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)

# Example input question
input_text = "다양한 엔진 부품을 정비할 때 필요한 게 뭐야"

# Tokenize the input question
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Generate response
outputs = model.generate(input_ids)

# Decode the generated response
generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

print("Generated response:", generated_text[0])