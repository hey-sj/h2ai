from flask import Flask, jsonify, request
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__, static_folder='../frontend/dist', static_url_path='/')
CORS(app)  # Enable CORS for all routes

# Sample API endpoint
@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify(message="test message")

# handle form submissions
@app.route('/api/submit', methods=['POST'])
def submit():
    data = request.json
    user_text = data.get('text', '')

    #process the user input
    response_text = f"You sent: {user_text}"

    return jsonify(response=response_text, success=True)

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