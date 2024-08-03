import os
from flask import Flask, request, jsonify
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Load models
qa_model = pipeline('question-answering', model='deepset/roberta-large-squad2', tokenizer='deepset/roberta-large-squad2')
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2', device='cpu')  # Force CPU usage

# Load your data
with open('data.txt', 'r') as file:
    data = file.read().split('\n')

# Create embeddings for your data
data_embeddings = embedder.encode(data, convert_to_tensor=True)

# Function to find the most relevant context
def find_relevant_context(question, data, data_embeddings):
    question_embedding = embedder.encode(question, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(question_embedding, data_embeddings)[0]
    top_score_idx = cos_scores.argmax().item()
    
    # Adjust context size if needed
    context_chunk = data[max(top_score_idx - 1, 0):top_score_idx + 1]
    return " ".join(context_chunk)

# Route for question-answering
@app.route('/ask', methods=['POST'])
def ask():
    question = request.json.get('question')
    if not question:
        return jsonify({'error': 'Question is required'}), 400

    context = find_relevant_context(question, data, data_embeddings)

    # Generate answer using the QA model
    result = qa_model(question=question, context=context)
    return jsonify({'answer': result['answer']})

# Route for health check
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)