from flask import Flask, jsonify, request
from flask_cors import CORS
import time
import os
from datetime import datetime
import pandas as pd
import torch
import gensim
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# Global Variables:
similar_symptoms = []
confirmed_symptoms = []
max_symptoms_to_ask = 20
asked_symptoms = set()
symptom_index = 0
similar_symptoms_length = 0
all_symptoms = []
mlb = ""
encoder = ""
scaler = ""
regression = ""
symptoms_transformed = ""
in_second_round = False



os.environ["TOKENIZERS_PARALLELISM"] = "false"

# This is symptoms and diagnoses
filepath = 'symbipredict_2022.csv'

# Load the data
csv_data = pd.read_csv(filepath)
csv_data.columns = csv_data.columns.str.replace('_', ' ')

# Extract symptoms and diagnoses
symptoms = csv_data.iloc[:, :-1]
prognosis = csv_data.iloc[:, -1]

# Prepare the csv data for Word2Vec
symptoms_list = symptoms.apply(lambda x: x.index[x == 1].tolist(), axis=1).tolist()

# Train the Word2Vec model
w2v = gensim.models.Word2Vec(
    symptoms_list,
    vector_size=100,
    window=10,
    min_count=2,
    workers=10,
)

# Load MedBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Charangan/MedBERT")
model = AutoModel.from_pretrained("Charangan/MedBERT")

# Detect useful words
def find_significant_words(input):
    words = input.split()
    significant_words = [word for word in words if word.lower() not in ENGLISH_STOP_WORDS]
    return significant_words


def get_medbert_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)



def find_similar_symptoms(user_input, symptoms, num_similar=10):
    significant_words = find_significant_words(user_input)
    symptom_embeddings = get_medbert_embeddings(symptoms)
    user_embeddings = get_medbert_embeddings(significant_words)
    similar_symptoms = set()

    for i, word in enumerate(user_embeddings):
        similarities = []
        for symptom in symptom_embeddings:
            similarity = cosine_similarity(word.reshape(1, -1), symptom.reshape(1, -1))[0][0]
            similarities.append(similarity)

        similar_indices = sorted(range(len(similarities)), key=lambda x: similarities[x], reverse=True)[:num_similar]
        for idx in similar_indices:
            similar_symptoms.add((symptoms[idx], similarities[idx]))

    similar_symptoms = sorted(similar_symptoms, key=lambda x: x[1], reverse=True)
    return [symptom for symptom, _ in similar_symptoms]


def getLikelyDiagnoses():
    global symptoms_transformed
    global similar_symptoms 
    global confirmed_symptoms
    global max_symptoms_to_ask
    global asked_symptoms
    global symptom_index
    global similar_symptoms_length
    global all_symptoms
    global mlb
    global encoder
    global scaler
    global regression
    mlb = MultiLabelBinarizer()
    mlb.fit(symptoms.apply(lambda x: x.index[x == 1].tolist(), axis=1).tolist())
    symptoms_binary = mlb.transform(symptoms.apply(lambda x: x.index[x == 1].tolist(), axis=1).tolist())

    # Convert symptoms_binary to DataFrame
    symptoms_binary_df = pd.DataFrame(symptoms_binary, columns=mlb.classes_)

    # Encode the features
    encoder = ColumnTransformer(transformers=[('onehot', OneHotEncoder(handle_unknown='ignore'), mlb.classes_)])
    symptoms_transformed = encoder.fit_transform(symptoms_binary_df)

    # Standardize the features
    scaler = StandardScaler()
    symptoms_transformed = scaler.fit_transform(symptoms_transformed)

    # Fit the regression model
    regression = LogisticRegression(max_iter=1000, class_weight='balanced')
    regression.fit(symptoms_transformed, prognosis)

    # Transform confirmed symptoms
    confirmed_symptoms_binary = mlb.transform([confirmed_symptoms])
    confirmed_symptoms_df = pd.DataFrame(confirmed_symptoms_binary, columns=mlb.classes_)
    confirmed_symptoms_transformed = encoder.transform(confirmed_symptoms_df)
    confirmed_symptoms_transformed = scaler.transform(confirmed_symptoms_transformed)

    # Calculate probabilities based on confirmed symptoms
    probabilities = regression.predict_proba(confirmed_symptoms_transformed)[0]
    new_diagnosis_probabilities = {diagnosis: prob for diagnosis, prob in zip(regression.classes_, probabilities) if prob > 0.00}

    high_prob_diag_string = ""
    sorted_diagnoses = sorted(new_diagnosis_probabilities.items(), key=lambda x: x[1], reverse=True)
    for diagnosis, probability in sorted_diagnoses:
        if probability > 0.05:
            high_prob_diag_string = high_prob_diag_string + f" {diagnosis}: {probability:.4f} "
        

    # extract a list of symptoms for diagnoses where probability is over 5%
    symptoms_for_diagnoses = {}
    for diagnosis, probability in sorted_diagnoses:
        if probability > 0.05:
            associated_symptoms = symptoms.columns[(symptoms[prognosis == diagnosis].sum() > 0).values].tolist()
            symptoms_for_diagnoses[diagnosis] = associated_symptoms

    # combine the list of symptoms for each diagnosis, unique values only
    all_symptoms = set()
    for symptoms_list in symptoms_for_diagnoses.values():
        all_symptoms.update(symptoms_list)

    # remove the confirmed symptoms from the list
    for symptom in confirmed_symptoms:
        if symptom in all_symptoms:
            all_symptoms.remove(symptom)
    
    return high_prob_diag_string

def getFinalDiagnoses():
    global confirmed_symptoms
    global symptom_index
    global all_symptoms
    global similar_symptoms 
    global max_symptoms_to_ask
    global asked_symptoms
    global similar_symptoms_length
    global scaler
    global regression
    global symptoms_transformed
    confirmed_symptoms_binary = mlb.transform([confirmed_symptoms])
    confirmed_symptoms_df = pd.DataFrame(confirmed_symptoms_binary, columns=mlb.classes_)
    confirmed_symptoms_transformed = encoder.transform(confirmed_symptoms_df)
    confirmed_symptoms_transformed = scaler.transform(confirmed_symptoms_transformed)

    # run regression model again with the new set of confirmed symptoms
    regression.fit(symptoms_transformed, prognosis)
    probabilities = regression.predict_proba(confirmed_symptoms_transformed)[0]

    new_diagnosis_probabilities = {diagnosis: prob for diagnosis, prob in zip(regression.classes_, probabilities) if prob > 0.00}

    high_prob_diag_string = ""
    sorted_diagnoses = sorted(new_diagnosis_probabilities.items(), key=lambda x: x[1], reverse=True)
    for diagnosis, probability in sorted_diagnoses:
        if probability > 0.05:
            print(f"{diagnosis}: {probability:.4f}")
            high_prob_diag_string = high_prob_diag_string + f" {diagnosis}: {probability:.4f} "
    return high_prob_diag_string

def prepare_second_round():
    global all_symptoms
    global similar_symptoms
    global symptom_index
    global in_second_round
    global asked_symptoms
    
    # Generate new list of symptoms to ask about based on most likely diagnoses
    # Convert all_symptoms set to a list for similar_symptoms
    similar_symptoms = list(all_symptoms)
    # Reset the symptom index to start fresh
    symptom_index = 0
    # Mark that we're in the second round
    in_second_round = True
    
    return similar_symptoms


# Initialize Flask app
app = Flask(__name__, static_folder='../frontend/dist', static_url_path='/')
CORS(app)  # Enable CORS for all routes

# Sample API endpoint
@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify(message="Hello from Flask!")

# Endpoint to process the submitted text
@app.route('/api/process-text', methods=['POST'])
def process_text():
    data = request.json
    user_text = data.get('text', '')
    global similar_symptoms 
    global confirmed_symptoms
    global max_symptoms_to_ask
    global asked_symptoms
    global symptom_index
    global similar_symptoms_length
    global in_second_round
    
    # Reset for a new session
    confirmed_symptoms = []
    asked_symptoms = set()
    symptom_index = 0
    in_second_round = False
    
    similar_symptoms = find_similar_symptoms(user_text, csv_data.columns[:-1].str.replace('_', ' ').tolist(), num_similar=20)
    symptom = similar_symptoms[symptom_index]
    similar_symptoms_length = len(similar_symptoms)

    question = f"Do you have '{symptom}'?"
    symptom_index = symptom_index + 1
    asked_symptoms.add(symptom)
    
    return jsonify(
        question=question,
        currentStep='question', # used to determine what page is rendered next
        success=True
    )

# New endpoint to start second round of questions
@app.route('/api/start-second-round', methods=['POST'])
def start_second_round():
    global symptom_index
    global similar_symptoms
    global in_second_round
    
    prepare_second_round()
    
    # Get the first symptom from the new list
    if similar_symptoms and len(similar_symptoms) > 0:
        symptom = similar_symptoms[symptom_index]
        symptom_index += 1
        question = f"Do you also have '{symptom}'?"
        
        return jsonify(
            question=question,
            currentStep='diagnosis',  # This will trigger the diagnosis question flow
            success=True
        )
    else:
        # If no more symptoms to ask about, go straight to final result
        result = getFinalDiagnoses()
        return jsonify(
            result=result,
            currentStep='result',
            success=True
        )

# Endpoint to handle the user's answer
@app.route('/api/answer', methods=['POST'])
def answer():
    data = request.json
    answer = data.get('answer', '')

    global similar_symptoms 
    global confirmed_symptoms
    global max_symptoms_to_ask
    global asked_symptoms
    global symptom_index
    global similar_symptoms_length
    global in_second_round
    
    # If the user said yes, add the symptom to confirmed symptoms
    if answer == 'yes':
        # Extract the symptom name from the question (which was the previous symptom)
        previous_symptom = similar_symptoms[symptom_index - 1]
        confirmed_symptoms.append(previous_symptom)
    
    # check for enough questions to show preliminary results
    if symptom_index >= max_symptoms_to_ask / 2 or symptom_index >= similar_symptoms_length:
        result = getLikelyDiagnoses()
        print("Preliminary diagnosis:", result)
        return jsonify(
            result=result,
            currentStep='midresult',  # Show midresult page
            success=True
        )
    
    # Otherwise, continue to the next question
    if symptom_index < similar_symptoms_length:
        symptom = similar_symptoms[symptom_index]
        question = f"Do you have '{symptom}'?"
        symptom_index = symptom_index + 1
        asked_symptoms.add(symptom)
        
        return jsonify(
            question=question,
            currentStep='question',
            success=True
        )
    else:
        # If we've run out of symptoms but haven't reached our threshold yet
        result = getLikelyDiagnoses()
        return jsonify(
            result=result,
            currentStep='midresult',
            success=True
        )


# Endpoint to handle the diagnosis answers (second round)
@app.route('/api/diagnose', methods=['POST'])
def diagnose():
    data = request.json
    answer = data.get('answer', '')

    global confirmed_symptoms
    global symptom_index
    global all_symptoms
    global similar_symptoms
    global in_second_round
    
    # If the user said yes, add the symptom to confirmed symptoms
    if answer == 'yes':
        # Extract the previous symptom
        previous_symptom = similar_symptoms[symptom_index - 1]
        if previous_symptom not in confirmed_symptoms:
            confirmed_symptoms.append(previous_symptom)
    
    # Check if we've reached the end of the second round
    if symptom_index >= max_symptoms_to_ask or symptom_index >= len(similar_symptoms):
        result = getFinalDiagnoses()
        print("Final diagnosis:", result)
        return jsonify(
            result=result,
            currentStep='result',  # Show final result page
            success=True
        )
    
    # Otherwise, continue with the next symptom
    if symptom_index < len(similar_symptoms):
        symptom = similar_symptoms[symptom_index]
        question = f"Do you also have '{symptom}'?"
        symptom_index = symptom_index + 1
        
        return jsonify(
            question=question,
            currentStep='diagnosis',  # Continue with diagnosis flow
            success=True
        )
    else:
        # If we've run out of symptoms
        result = getFinalDiagnoses()
        return jsonify(
            result=result,
            currentStep='result',
            success=True
        )

# Route to serve React app
@app.route('/')
def serve():
    return app.send_static_file('index.html')

# Catch-all route to handle React router
@app.route('/<path:path>')
def catch_all(path):
    try:
        return app.send_static_file(path)
    except:
        return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(debug=True)