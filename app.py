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

# Global variable for the refinement URL
REFINEMENT_API_URL = os.getenv('REFINEMENT_API_URL', 'http://167.71.231.121:8080/api/ai/generate')

# Load models
qa_model = pipeline('question-answering', model='deepset/roberta-large-squad2', tokenizer='deepset/roberta-large-squad2')
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2', device='cpu')  # Force CPU usage

# Function to load data and compute embeddings
def load_data_and_compute_embeddings(file_path):
    if not os.path.exists(file_path):
        open(file_path, 'w').close()  # Create file if it doesn't exist
    with open(file_path, 'r') as file:
        data = file.read().split('\n')
    data_embeddings = embedder.encode(data, convert_to_tensor=True)
    return data, data_embeddings

# Initial load
data, data_embeddings = load_data_and_compute_embeddings('data.txt')
data_qa, data_qa_embeddings = load_data_and_compute_embeddings('qa.txt')

def find_relevant_context_top(question, data, data_embeddings):
    question_embedding = embedder.encode(question, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(question_embedding, data_embeddings)[0]
    top_score_idx = cos_scores.argmax().item()  # Get the index of the highest score
    
    # Get the top paragraph
    context_chunk = data[top_score_idx]
    # find also similarity score from it 
    score = cos_scores[top_score_idx].item()
    return json.dumps({'context': context_chunk, 'score': score})

# Function to find the most relevant context
def find_relevant_context(question, data, data_embeddings):
    question_embedding = embedder.encode(question, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(question_embedding, data_embeddings)[0]
    top_score_idx = cos_scores.argsort(descending=True)[:2].tolist()  # Get indices of top 2 scores
    
    # Get top two paragraphs
    context_chunks = [data[idx] for idx in top_score_idx]
    return " ".join(context_chunks)

def refine_answer(question, response):
    prompt = (f"Here is given a question and its answer : Question : {question} ? , "
              f"Answer : {response} return a jsonObject like this {{'answer' : ''}} "
              f"that refines the answer according to question in a human-like language "
              f"and if you don't get answer in context just return that you don't have information")

    try:
        response = requests.post(REFINEMENT_API_URL, data={'prompt': prompt})
        response.raise_for_status()  # Raise an exception for bad status codes
        
        response_text = response.text
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
       
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
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
    return jsonify({'answer': answer})

@app.route('/ask/refined', methods=['POST'])
def ask_refined():
    question = request.json.get('question')
    if not question:
        return jsonify({'error': 'Question is required'}), 400

    context = find_relevant_context(question, data, data_embeddings)

    # Generate answer using the question-answering model
    result = qa_model(question=question, context=context)
    answer = result['answer']
    refined_response = refine_answer(question, answer)
    return jsonify(refined_response)

@app.route('/ask/context/refined', methods=['POST'])
def ask_refined_context():
    question = request.json.get('question')
    if not question:
        return jsonify({'error': 'Question is required'}), 400

    context = find_relevant_context_top(question, data, data_embeddings)
    context = json.loads(context)
    score = context['score']
    if(score < 0.25):
        return jsonify({'answer': '', 'score': score})
    refined_response = refine_answer(question, context['context'])
    return jsonify(refined_response)

# return top context from qa file 
@app.route('/ask_top', methods=['POST'])
def ask_top():
    question = request.json.get('question')
    if not question:
        return jsonify({'error': 'Question is required'}), 400
    
    context = find_relevant_context_top(question, data_qa, data_qa_embeddings)

    # answer is written after | sign in context = {"context" : "",score:"")
    context = json.loads(context)
    answer = context['context'].split('|')
    return jsonify({'answer': answer[1], 'score': context['score']})

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
    data, data_embeddings = load_data_and_compute_embeddings('data.txt')

    return jsonify({'status': 'Data added and embeddings updated'})

@app.route('/add_data_qa', methods=['POST'])
def add_data_qa():
    new_data_qa = request.json.get('data_qa')
    if not new_data_qa:
        return jsonify({'error': 'Data is required'}), 400

    with open('qa.txt', 'a') as file:
        file.write('\n' + new_data_qa)
    
    # Recompute embeddings
    global data_qa, data_qa_embeddings
    data_qa, data_qa_embeddings = load_data_and_compute_embeddings('qa.txt')

    return jsonify({'status': 'Data added and embeddings updated'})

@app.route('/clear_data', methods=['POST'])
def clear_data():
    open('data.txt', 'w').close()
    return jsonify({'status': 'All data cleared and embeddings updated'})

@app.route('/clear_data_qa', methods=['POST'])
def clear_data_qa():
    open('qa.txt', 'w').close()
    return jsonify({'status': 'All data cleared and embeddings updated'})

# Route to change the refinement URL
@app.route('/set_refinement_url', methods=['POST'])
def set_refinement_url():
    global REFINEMENT_API_URL
    new_url = request.json.get('url')
    if not new_url:
        return jsonify({'error': 'URL is required'}), 400

    REFINEMENT_API_URL = new_url
    return jsonify({'status': 'Refinement URL updated', 'new_url': REFINEMENT_API_URL})

# Route for health check
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
