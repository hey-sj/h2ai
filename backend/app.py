from flask import Flask, jsonify, request
from flask_cors import CORS
import time
import os
from datetime import datetime

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
    

    # In a real application, you would do some analysis here
    # For this example, let's just simulate processing time
    time.sleep(1)  # Simulate processing time
    print(datetime.now())
    # Generate a question based on the text
    # In a real app, this would be more sophisticated
    question = f" '{user_text[:30]}...' contain important information?"
    
    return jsonify(
        question=question,
        currentStep='question', # used to determine what page is rendered next
        success=True
    )

# Endpoint to handle the user's answer
@app.route('/api/answer', methods=['POST'])
def answer():
    data = request.json
    user_text = data.get('text', '')
    question = data.get('question', '')
    answer = data.get('answer', '')
    
    # In a real application, you would do further processing based on the answer
    # For this example, we'll just simulate processing time
    time.sleep(1)  # Simulate processing time
    
    # Generate a result based on the answer
    if answer == 'yes':
        result = "Thank you for confirming! We've recorded that this text contains important information."
        return jsonify(
        result=result,
        currentStep='result', # This is how we determine what page is rendered
        success=True
    )

    else:
        result = "Thanks for your input! We've noted that this text doesn't contain important information."
        return jsonify(
        result=result,
        currentStep='question', # This is how we determine what page is rendered
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