import os
import pandas as pd
import torch
import gensim
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS



os.environ["TOKENIZERS_PARALLELISM"] = "false"

# This is symptoms and diagnoses
filepath = 'symbipredict_2022.csv'

# Load the data
data = pd.read_csv(filepath)
data.columns = data.columns.str.replace('_', ' ')

# Extract symptoms and diagnoses
symptoms = data.iloc[:, :-1]
prognosis = data.iloc[:, -1]

# Prepare the data for Word2Vec
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






# Main function
if __name__ == "__main__":
    user_input = input("Enter your symptoms: ")
    similar_symptoms = find_similar_symptoms(user_input, data.columns[:-1].str.replace('_', ' ').tolist(), num_similar=20)

    confirmed_symptoms = []
    max_symptoms_to_ask = 20  # Limit the number of symptoms to ask about
    asked_symptoms = set()  # Track asked symptoms to avoid duplicates
    for i, symptom in enumerate(similar_symptoms):
        if i >= max_symptoms_to_ask:
            break
        if symptom in asked_symptoms:
            continue
        asked_symptoms.add(symptom)
        response = input(f"Do you have {symptom}? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            confirmed_symptoms.append(symptom)

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

    print("Probabilities of different diagnoses based on confirmed symptoms:")
    sorted_diagnoses = sorted(new_diagnosis_probabilities.items(), key=lambda x: x[1], reverse=True)
    for diagnosis, probability in sorted_diagnoses:
        if probability > 0.05:
            print(f"{diagnosis}: {probability:.4f}")

    # extract a list of symptoms for diagnoses where probability is over 10%
    symptoms_for_diagnoses = {}
    for diagnosis, probability in sorted_diagnoses:
        if probability > 0.05:
            associated_symptoms = symptoms.columns[(symptoms[prognosis == diagnosis].sum() > 0).values].tolist()
            symptoms_for_diagnoses[diagnosis] = associated_symptoms

    # debug
    print("symptoms for dianoses", symptoms_for_diagnoses)

    # combine the list of symptoms for each diagnosis, unique values only
    all_symptoms = set()
    for symptoms_list in symptoms_for_diagnoses.values():
        all_symptoms.update(symptoms_list)

    # debug
    print("all symptoms", all_symptoms)

    # remove the confirmed symptoms from the list
    for symptom in confirmed_symptoms:
        if symptom in all_symptoms:
            all_symptoms.remove(symptom)



    # ask the user about the remaining symptoms
    for symptom in all_symptoms:
        response = input(f"Do you have {symptom}? (yes/no): ").strip().lower()
        if 'y' in response:
            if symptom not in confirmed_symptoms:
                confirmed_symptoms.append(symptom)

    # re-calculate the probabilities based on the new set of confirmed symptoms
    confirmed_symptoms_binary = mlb.transform([confirmed_symptoms])
    confirmed_symptoms_df = pd.DataFrame(confirmed_symptoms_binary, columns=mlb.classes_)
    confirmed_symptoms_transformed = encoder.transform(confirmed_symptoms_df)
    confirmed_symptoms_transformed = scaler.transform(confirmed_symptoms_transformed)

    # run regression model again with the new set of confirmed symptoms

    regression.fit(symptoms_transformed, prognosis)
    probabilities = regression.predict_proba(confirmed_symptoms_transformed)[0]

    new_diagnosis_probabilities = {diagnosis: prob for diagnosis, prob in zip(regression.classes_, probabilities) if prob > 0.00}

    print("Probabilities of different diagnoses based on confirmed symptoms:")
    sorted_diagnoses = sorted(new_diagnosis_probabilities.items(), key=lambda x: x[1], reverse=True)
    for diagnosis, probability in sorted_diagnoses:
        if probability > 0.05:
            print(f"{diagnosis}: {probability:.4f}")
