import os
from flask import Flask, request, jsonify
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAIGPT
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Load your model
model_name = "EleutherAI/gpt-neo-2.7B"
generator = pipeline('text-generation', model=model_name)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load your data
with open('data.txt', 'r') as file:
    data = file.read().splitlines()

# Create embeddings for your data
data_embeddings = embedder.encode(data, convert_to_tensor=True)

# Define a prompt template
prompt_template = PromptTemplate(input_variables=["context", "question"], template="{context}\n\nQuestion: {question}\nAnswer:")

# Define a custom LLM using the generator
class CustomLLM(OpenAIGPT):
    def __init__(self, generator):
        self.generator = generator

    def __call__(self, prompt):
        response = self.generator(prompt, max_length=150, num_return_sequences=1)
        return response[0]['generated_text']

llm = CustomLLM(generator)

# Create a chain
chain = LLMChain(llm=llm, prompt_template=prompt_template)

# Function to find the most relevant context
def find_relevant_context(question, data, data_embeddings):
    question_embedding = embedder.encode(question, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(question_embedding, data_embeddings)[0]
    top_score_idx = cos_scores.argmax().item()
    return data[top_score_idx]

# Route for question-answering
@app.route('/ask', methods=['POST'])
def ask():
    question = request.json.get('question')
    if not question:
        return jsonify({'error': 'Question is required'}), 400

    context = find_relevant_context(question, data, data_embeddings)
    response = chain.run(context=context, question=question)
    return jsonify({'answer': response})

# Route for health check
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
