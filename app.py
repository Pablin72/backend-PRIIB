from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess

app = Flask(__name__)
CORS(app)

@app.route('/example', methods=['POST'])
def execute_script():
  data = request.json
  name = data.get('name')
  age = data.get('age')

  if not name:
    return jsonify({"error": "Name is required"}), 400

  # Construct the command to run the script
  command = ['python', 'engine.py', '-n', name]
  if age:
    command.extend(['-a', str(age)])

  try:
    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    output = result.stdout
  except subprocess.CalledProcessError as e:
    return jsonify({"error": str(e)}), 500

  return jsonify({"output": output})

@app.route('/')
def hello():
  return 'Hello, world!'

if __name__ == '__main__':
  app.run()