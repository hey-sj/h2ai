import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
file_path = "symbipredict_2022.csv"
df = pd.read_csv(file_path)

# Extract symptom names (assuming symptoms are column names except the last column)
symptom_columns = df.columns[:-1].tolist()

# Load SBERT model
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight & fast

# Encode all symptoms into SBERT embeddings
symptom_embeddings = sbert_model.encode(symptom_columns)


# Function to find similar symptoms through semantics
def match_symptom_sbert(user_input, symptom_embeddings, symptoms):
    user_embedding = sbert_model.encode([user_input])  # Convert user sentence to vector
    top_n = 10
    # Compute similarity scores
    similarities = cosine_similarity(user_embedding, symptom_embeddings)[0]

    # Sort symptoms by similarity (highest first)
    sorted_indices = np.argsort(similarities)[::-1]  # Get indices of top matches
    sorted_similarities = similarities[sorted_indices]
    sorted_symptoms = [symptoms[i] for i in sorted_indices]

    # Filter symptoms above threshold and return top N matches into a list
    matched_symptoms = [
        (symptom, round(score, 2))  # Modify this line to change the format of the output
        for symptom, score in zip(sorted_symptoms, sorted_similarities)
    ]

    return matched_symptoms[:top_n] if matched_symptoms else [("No strong match found", 0)]


# Example input
user_sentence = "I have intense fatigue and extreme muscle pain, and my head feels hot."
matched_symptoms = match_symptom_sbert(user_sentence, symptom_embeddings, symptom_columns)
print("Matched Symptoms:", matched_symptoms)

# import numpy as np
# import pandas as pd
# import re
# import nltk
# from gensim.models import Word2Vec
# from sklearn.metrics.pairwise import cosine_similarity
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
#
# # Download necessary NLTK resources
# # nltk.download('punkt')
# # nltk.download('stopwords')
# # nltk.download('wordnet')
#
# # Load dataset
# file_path = "symbipredict_2022.csv"
# df = pd.read_csv(file_path)
#
# # Initialize stopwords and lemmatizer
# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()
#
# # Tokenization function with stopword removal and lemmatization
# def clean_tokenizer(text):
#     tokens = word_tokenize(text.lower())  # Convert to lowercase and tokenize
#     tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]  # Remove stopwords and non-alphanumeric
#     return tokens
#
# # Extract symptom names and tokenize them
# symptom_columns = df.columns[:-1].tolist()
# symptom_tokens = [clean_tokenizer(symptom.replace("_", " ")) for symptom in symptom_columns]
#
# # Train Word2Vec model
# word2vec_model = Word2Vec(vector_size=100, window=5, min_count=1, workers=4)
# word2vec_model.build_vocab(symptom_tokens)
# word2vec_model.train(symptom_tokens, total_examples=len(symptom_tokens), epochs=10)
# similarity = word2vec_model.similarity("Fever", "heavy fever")
# print(similarity)
# # Convert symptoms to Word2Vec vectors
# def get_symptom_vectors(symptoms, model):
#     symptom_vectors = {}
#     for symptom in symptoms:
#         words = clean_tokenizer(symptom)  # Clean tokens
#         vectors = [model.wv[word] for word in words if word in model.wv]
#         if vectors:
#             symptom_vectors[symptom] = np.mean(vectors, axis=0)  # Aggregate vectors
#     return symptom_vectors
#
# symptom_vectors = get_symptom_vectors(symptom_columns, word2vec_model)
#
# # Function to process user input and return multiple matching symptoms
# # def match_symptom(user_input, model, symptom_vectors, threshold=0.3):
# #     words = clean_tokenizer(user_input)  # Clean user input
# #     vectors = [model.wv[word] for word in words if word in model.wv]
# #
# #     if not vectors:
# #         return ["No relevant symptoms found."]
# #
# #     sentence_vector = np.mean(vectors, axis=0)  # Aggregate sentence vector
# #
# #     # Compute similarity with predefined symptoms
# #     similarities = {
# #         symptom: cosine_similarity([sentence_vector], [vector])[0][0]
# #         for symptom, vector in symptom_vectors.items()
# #     }
# #
# #     # Sort symptoms by similarity (descending order)
# #     sorted_symptoms = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
# #
# #     # Filter symptoms above the threshold
# #     matched_symptoms = [symptom for symptom, similarity in sorted_symptoms]
# #
# #     return matched_symptoms if matched_symptoms else ["No strong match found."]
#
# def match_symptom(user_input, model, symptom_vectors, threshold=0.1):
#     words = clean_tokenizer(user_input)  # Tokenize user input
#     print("Processed Tokens:", words)  # Debugging: Show cleaned tokens
#
#     # Step 1: Exact match check (Improved)
#     exact_matches = [
#         symptom for symptom in symptom_vectors
#         if any(word in clean_tokenizer(symptom) for word in words)
#     ]
#     print("Exact Matches:", exact_matches)  # Debugging: Show exact matches
#
#     # Step 2: Remove exact matches from similarity check
#     remaining_words = [
#         word for word in words
#         if not any(word in clean_tokenizer(symptom) for symptom in exact_matches)
#     ]
#
#     vectors = [model.wv[word] for word in remaining_words if word in model.wv]
#
#     if not vectors and not exact_matches:
#         return ["No relevant symptoms found."]
#
#     # Step 3: Similarity matching for remaining words
#     sentence_vector = np.mean(vectors, axis=0) if vectors else None  # Aggregate sentence vector
#
#     if sentence_vector is not None:
#         similarities = {
#             symptom: cosine_similarity([sentence_vector], [vector])[0][0]
#             for symptom, vector in symptom_vectors.items()
#         }
#
#         # Sort symptoms by similarity (descending order)
#         sorted_symptoms = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
#
#         # Filter symptoms above the threshold
#         similar_matches = [symptom for symptom, similarity in sorted_symptoms if similarity >= threshold]
#     else:
#         similar_matches = []
#
#     # Step 4: Combine exact matches and similarity-based matches
#     final_matches = list(set(exact_matches + similar_matches))  # Remove duplicates
#
#     return final_matches if final_matches else ["No strong match found."]
#
#
# # Example user input
# user_sentence = "I am experiencing extreme tiredness and a bad cough and fever."
# matched_symptoms = match_symptom(user_sentence, word2vec_model, symptom_vectors)
# print("Matched Symptoms:", matched_symptoms)

import numpy as np
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import torch
