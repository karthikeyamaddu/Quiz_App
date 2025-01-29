from flask import Flask, request, jsonify
import os
import fitz  # PyMuPDF
import google.generativeai as genai
from dotenv import load_dotenv
from flask_cors import CORS
import re
from pymongo import MongoClient
from datetime import datetime, timezone  # Import timezone
from bson import ObjectId

# Load environment variables
load_dotenv()

app = Flask(__name__)

# CORS setup to allow requests from any origin
CORS(app)

# MongoDB Atlas Configuration
MONGODB_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGODB_URI)
db = client.quizapp

# Collections
quizzes = db.quizzes
attempts = db.attempts
results = db.results

# Load Google Gemini API key
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("Error: GEMINI_API_KEY is missing from environment variables!")

# Configure Google Gemini API
genai.configure(api_key=gemini_api_key)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'Empty file selected'}), 400
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are supported'}), 400

        pdf_text = extract_text_from_pdf(file)
        if not pdf_text.strip():
            return jsonify({'error': 'No text extracted from the PDF'}), 400

        quiz_questions = generate_quiz_from_text(pdf_text)
        if not quiz_questions:
            return jsonify({'error': 'Failed to generate quiz questions'}), 500

        # Store quiz in MongoDB
        quiz_data = {
            'filename': file.filename,
            'questions': quiz_questions,
            'created_at': datetime.now(timezone.utc),  # Use timezone-aware datetime
            'times_taken': 0,
            'pdf_text': pdf_text  # Store the PDF text for reference
        }
        
        quiz_id = quizzes.insert_one(quiz_data).inserted_id
        
        return jsonify({
            'quizId': str(quiz_id),
            'quizQuestions': quiz_questions
        })

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/submit-result', methods=['POST'])
def submit_result():
    try:
        data = request.json
        quiz_id = data.get('quizId')
        user_answers = data.get('selectedAnswers')
        final_score = data.get('score')
        total_questions = data.get('totalQuestions')

        if not all([quiz_id, user_answers, final_score is not None]):
            return jsonify({'error': 'Missing required fields'}), 400

        # Calculate percentage
        score_percentage = (final_score / total_questions) * 100

        # Store result in MongoDB
        result_data = {
            'quiz_id': ObjectId(quiz_id),
            'user_answers': user_answers,
            'score': final_score,
            'score_percentage': score_percentage,
            'total_questions': total_questions,
            'timestamp': datetime.now(timezone.utc)  # Use timezone-aware datetime
        }
        
        result_id = results.insert_one(result_data).inserted_id

        # Update quiz statistics
        quizzes.update_one(
            {'_id': ObjectId(quiz_id)},
            {
                '$inc': {'times_taken': 1},
                '$push': {
                    'scores': score_percentage
                }
            }
        )

        # Calculate and store analytics
        all_results = list(results.find({'quiz_id': ObjectId(quiz_id)}))
        avg_score = sum(result['score_percentage'] for result in all_results) / len(all_results) if all_results else 0
        
        analytics_data = {
            'quiz_id': ObjectId(quiz_id),
            'total_attempts': len(all_results),
            'average_score': avg_score,
            'last_updated': datetime.now(timezone.utc)  # Use timezone-aware datetime
        }
        
        db.analytics.update_one(
            {'quiz_id': ObjectId(quiz_id)},
            {'$set': analytics_data},
            upsert=True
        )

        return jsonify({
            'resultId': str(result_id),
            'message': 'Result submitted successfully',
            'analytics': {
                'average_score': avg_score,
                'total_attempts': len(all_results)
            }
        })

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/quiz-stats/<quiz_id>', methods=['GET'])
def get_quiz_stats(quiz_id):
    try:
        quiz = quizzes.find_one({'_id': ObjectId(quiz_id)})
        if not quiz:
            return jsonify({'error': 'Quiz not found'}), 404

        # Get all results for this quiz
        quiz_results = list(results.find({'quiz_id': ObjectId(quiz_id)}))
        
        # Calculate statistics
        total_attempts = len(quiz_results)
        if total_attempts > 0:
            average_score = sum(result['score_percentage'] for result in quiz_results) / total_attempts
            highest_score = max(result['score_percentage'] for result in quiz_results)
            lowest_score = min(result['score_percentage'] for result in quiz_results)
        else:
            average_score = highest_score = lowest_score = 0

        return jsonify({
            'quiz_id': quiz_id,
            'filename': quiz['filename'],
            'times_taken': total_attempts,
            'average_score': average_score,
            'highest_score': highest_score,
            'lowest_score': lowest_score,
            'created_at': quiz['created_at'].isoformat()
        })

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

# Helper functions
def extract_text_from_pdf(pdf_file):
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = []
    for page in pdf_document:
        text.append(page.get_text("text"))
    return " ".join(text)

def generate_quiz_from_text(text):
    max_chars = 5000
    text = text[:max_chars]

    prompt = f"""
    Based on the following text, generate 5 multiple-choice questions. 
    Each question should include:
    1. A clear question
    2. Four answer choices (A, B, C, D)
    3. The correct answer

    Format each question as follows:
    Q: [Question text]
    A) [Option A]
    B) [Option B]
    C) [Option C]
    D) [Option D]
    Correct: [A/B/C/D]

    Text: {text}
    """

    try:
        # Use the gemini-2.0-flash-exp model here
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        response = model.generate_content(prompt)

        if not response or not response.text:
            raise Exception("Gemini API returned an empty response")

        return parse_quiz_response(response.text)

    except Exception as e:
        print(f"Debug: Error calling Gemini API - {str(e)}")
        return []

def parse_quiz_response(response_text):
    questions = []
    current_question = None

    lines = response_text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if re.match(r"^Q:", line):
            if current_question:
                questions.append(current_question)
            current_question = {"question": line[2:].strip(), "choices": [], "correctAnswer": None}

        elif re.match(r"^[A-D]\)", line):
            if current_question:
                current_question["choices"].append(line[2:].strip())

        elif re.match(r"^Correct:", line):
            if current_question:
                correct_letter = line.split(":")[-1].strip()
                correct_index = ord(correct_letter) - ord('A')
                if 0 <= correct_index < len(current_question["choices"]):
                    current_question["correctAnswer"] = current_question["choices"][correct_index]

    if current_question:
        questions.append(current_question)

    return [q for q in questions if q["choices"] and q["correctAnswer"]]

if __name__ == '__main__':
    app.run(debug=True, port=5000)
