import os
from flask import Flask, request, jsonify
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from flask_cors import CORS
import torch
import requests
import json

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

# Load models
qa_model = pipeline('question-answering', model='deepset/roberta-large-squad2', tokenizer='deepset/roberta-large-squad2')
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2', device='cpu')  # Force CPU usage

# Function to load data and compute embeddings
def load_data_and_compute_embeddings():
    with open('data.txt', 'r') as file:
        data = file.read().split('\n')
    data_embeddings = embedder.encode(data, convert_to_tensor=True)
    return data, data_embeddings

# Initial load
data, data_embeddings = load_data_and_compute_embeddings()

# Function to find the most relevant context
def find_relevant_context(question, data, data_embeddings):
    question_embedding = embedder.encode(question, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(question_embedding, data_embeddings)[0]
    top_score_idx = cos_scores.argsort(descending=True)[:2].tolist()  # Get indices of top 2 scores
    
    # Get top two paragraphs
    context_chunks = [data[idx] for idx in top_score_idx]
    return " ".join(context_chunks)

def refine_answer(question, response):
    refinement_url = 'http://167.71.231.121:8080/api/ai/generate'
    prompt = f'Here is given a question and its answer : Question : {question} ? , Answer : {response}  return a jsonObject  like this {{\'answer\' : \'\'}}  , that refines the answer according to question in a human like language and if you don\'t know answer just return that you don\'t have information'
    
    try:
        response = requests.post(refinement_url, data={'prompt': prompt})
        response.raise_for_status()  # Raise an exception for bad status codes
        
        response_text = response.text
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
       
        if (start_idx != -1 and end_idx != -1 and start_idx < end_idx):
            json_str = response_text[start_idx:end_idx]
            json_data = json.loads(json_str)
            return json_data
        else:
            return {'error': 'Invalid response format'}
    except requests.RequestException as e:
        return {'error': f'Request to refinement API failed: {str(e)}'}
    except ValueError as e:
        return {'error': f'Failed to parse refined response: {str(e)}'}

# Route for question-answering
@app.route('/ask', methods=['POST'])
def ask():
    question = request.json.get('question')
    if not question:
        return jsonify({'error': 'Question is required'}), 400

    context = find_relevant_context(question, data, data_embeddings)

    # Generate answer using the question-answering model
    result = qa_model(question=question, context=context)
    answer = result['answer']
    # refined_response = refine_answer(question, answer)
    return jsonify({'answer': answer})

@app.route('/ask/refined', methods=['POST'])
def askk():
    question = request.json.get('question')
    if not question:
        return jsonify({'error': 'Question is required'}), 400

    context = find_relevant_context(question, data, data_embeddings)

    # Generate answer using the question-answering model
    result = qa_model(question=question, context=context)
    answer = result['answer']
    refined_response = refine_answer(question, answer)
    return jsonify(refined_response)

# Route for adding data to data.txt
@app.route('/add_data', methods=['POST'])
def add_data():
    new_data = request.json.get('data')
    if not new_data:
        return jsonify({'error': 'Data is required'}), 400

    with open('data.txt', 'a') as file:
        file.write('\n' + new_data)
    
    # Recompute embeddings
    global data, data_embeddings
    data, data_embeddings = load_data_and_compute_embeddings()

    return jsonify({'status': 'Data added and embeddings updated'})

@app.route('/clear_data', methods=['POST'])
def clear_data():
    open('data.txt', 'w').close()
    return jsonify({'status': 'All data cleared'})

# Route for health check
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
