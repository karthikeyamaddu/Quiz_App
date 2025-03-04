import random
from flask import Flask, request, jsonify
import os
import fitz  # PyMuPDF
import google.generativeai as genai
from dotenv import load_dotenv
from flask_cors import CORS
import re
from pymongo import MongoClient
from datetime import datetime, timezone
from bson import ObjectId

# Load environment variables
load_dotenv()

app = Flask(__name__)

# CORS setup for local frontend (React at http://localhost:3000)
CORS(app)

# MongoDB Configuration (using local or Atlas URI from .env)
MONGODB_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGODB_URI)
db = client.quiz_app  # Simplified database name
quizzes_collection = db.quizzes
results_collection = db.results

# Google Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from environment variables!")
genai.configure(api_key=GEMINI_API_KEY)

# Helper function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = " ".join(page.get_text("text") for page in pdf_document)
        pdf_document.close()
        return text.strip()
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")

# Helper function to generate quiz questions with Gemini API
def generate_quiz_from_text(text):
    max_chars = 5000
    truncated_text = text[:max_chars]
    question_count = min(5, max(1, len(truncated_text.split()) // 50))  # Dynamic question count

    prompt = f"""
    Based on the following text, generate {question_count} multiple-choice questions.
    Each question should include:
    1. A clear question
    2. Four answer choices (A, B, C, D)
    3. The correct answer
    4. A brief explanation

    Format each question as follows:
    Q: [Question text]
    A) [Option A]
    B) [Option B]
    C) [Option C]
    D) [Option D]
    Correct: [A/B/C/D]
    Explanation: [Brief explanation]

    Text: {truncated_text}
    """

    try:
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        response = model.generate_content(prompt)
        if not response or not response.text:
            raise Exception("Gemini API returned an empty response")
        questions = parse_quiz_response(response.text)
        return questions if questions else fallback_quiz_generation(truncated_text)
    except Exception as e:
        print(f"Error with Gemini API: {str(e)}")
        return fallback_quiz_generation(truncated_text)

# Fallback quiz generation if Gemini fails
def fallback_quiz_generation(text):
    sentences = [s.strip() for s in text.split('.') if s.strip() and len(s.split()) > 5]
    questions = []
    for _ in range(min(3, len(sentences))):
        if not sentences:
            break
        sentence = sentences.pop(0)
        words = sentence.split()
        correct_answer = random.choice(words[1:-1])  # Avoid first/last word
        choices = [correct_answer]
        while len(choices) < 4 and words:
            wrong = random.choice([w for w in words if w not in choices])
            choices.append(wrong)
        random.shuffle(choices)
        questions.append({
            'question': f"What is a key term in: '{sentence}'?",
            'choices': choices,
            'correctAnswer': correct_answer,
            'explanation': f"'{correct_answer}' is a notable term in this context."
        })
    return questions

# Parse Gemini API response
def parse_quiz_response(response_text):
    questions = []
    current_question = None

    for line in response_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if re.match(r"^Q:", line):
            if current_question and current_question["choices"] and current_question["correctAnswer"]:
                questions.append(current_question)
            current_question = {"question": line[2:].strip(), "choices": [], "correctAnswer": None, "explanation": ""}
        elif re.match(r"^[A-D]\)", line):
            if current_question:
                current_question["choices"].append(line[2:].strip())
        elif re.match(r"^Correct:", line):
            if current_question:
                correct_letter = line.split(":")[-1].strip()
                correct_index = ord(correct_letter) - ord('A')
                if 0 <= correct_index < len(current_question["choices"]):
                    current_question["correctAnswer"] = current_question["choices"][correct_index]
        elif re.match(r"^Explanation:", line):
            if current_question:
                current_question["explanation"] = line.split(":", 1)[-1].strip()

    if current_question and current_question["choices"] and current_question["correctAnswer"]:
        questions.append(current_question)

    return questions

# Upload endpoint
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are supported'}), 400

        pdf_text = extract_text_from_pdf(file)
        if not pdf_text:
            return jsonify({'error': 'No text extracted from the PDF'}), 400

        quiz_questions = generate_quiz_from_text(pdf_text)
        if not quiz_questions:
            return jsonify({'error': 'Failed to generate quiz questions'}), 500

        quiz_data = {
            'filename': file.filename,
            'questions': quiz_questions,
            'created_at': datetime.now(timezone.utc),
            'times_taken': 0,
            'pdf_text': pdf_text
        }
        result = quizzes_collection.insert_one(quiz_data)
        return jsonify({
            'quizId': str(result.inserted_id),
            'quizQuestions': quiz_questions
        }), 200

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

# Submit result endpoint
@app.route('/submit-result', methods=['POST'])
def submit_result():
    try:
        data = request.get_json()
        quiz_id = data.get('quizId')
        selected_answers = data.get('selectedAnswers')
        score = data.get('score')
        total_questions = data.get('totalQuestions')
        time_spent = data.get('timeSpent')

        if not all([quiz_id, selected_answers, isinstance(score, (int, float)), isinstance(total_questions, int), isinstance(time_spent, (int, float))]):
            return jsonify({'error': 'Missing or invalid required fields'}), 400

        quiz = quizzes_collection.find_one({'_id': ObjectId(quiz_id)})
        if not quiz:
            return jsonify({'error': 'Quiz not found'}), 404

        score_percentage = (score / total_questions) * 100
        result_data = {
            'quizId': ObjectId(quiz_id),
            'selectedAnswers': selected_answers,
            'score': score,
            'scorePercentage': score_percentage,
            'totalQuestions': total_questions,
            'timeSpent': time_spent,
            'submittedAt': datetime.now(timezone.utc)
        }
        result_id = results_collection.insert_one(result_data).inserted_id

        quizzes_collection.update_one(
            {'_id': ObjectId(quiz_id)},
            {'$inc': {'times_taken': 1}}
        )

        return jsonify({'message': 'Result submitted', 'resultId': str(result_id)}), 200

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

# Quiz stats endpoint
@app.route('/quiz-stats/<quiz_id>', methods=['GET'])
def get_quiz_stats(quiz_id):
    try:
        quiz = quizzes_collection.find_one({'_id': ObjectId(quiz_id)})
        if not quiz:
            return jsonify({'error': 'Quiz not found'}), 404

        quiz_results = list(results_collection.find({'quizId': ObjectId(quiz_id)}))
        times_taken = len(quiz_results)
        if times_taken > 0:
            scores = [r['scorePercentage'] for r in quiz_results]
            average_score = sum(scores) / times_taken
            highest_score = max(scores)
        else:
            average_score = highest_score = 0

        return jsonify({
            'times_taken': times_taken,
            'average_score': round(average_score, 1),
            'highest_score': round(highest_score, 1)
        }), 200

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

if __name__ == '__main__':
    # Create indexes for performance
    quizzes_collection.create_index([('created_at', -1)])
    results_collection.create_index([('quizId', 1), ('submittedAt', -1)])
    app.run(host='127.0.0.1', port=5000, debug=True)