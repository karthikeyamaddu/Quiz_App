# Quiz Generator Backend 🎯

A powerful Flask-based backend API that converts PDF documents into interactive quiz questions using Google Gemini AI. Perfect for creating study materials and educational content.

## 📋 Features

- **PDF Upload & Processing**: Upload PDF files and automatically extract text content
- **AI-Powered Quiz Generation**: Uses Google Gemini AI to generate intelligent multiple-choice questions
- **Quiz Storage**: Store and manage quizzes in MongoDB
- **Results Tracking**: Track user quiz attempts, scores, and performance metrics
- **Statistics API**: Get detailed quiz statistics including average scores and top scores
- **CORS Enabled**: Ready for integration with React or any frontend framework
- **Fallback Generation**: Graceful fallback mechanism if AI API fails
- **Performance Optimized**: Database indexes for fast queries

## 🛠️ Technology Stack

- **Framework**: Flask 3.1.0
- **Database**: MongoDB
- **AI Engine**: Google Gemini API
- **PDF Processing**: PyMuPDF
- **Validation**: Pydantic
- **Server**: Gunicorn
- **Other**: Flask-CORS, python-dotenv

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- MongoDB (local or cloud)
- Google Gemini API key
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/karthikeyamaddu/Quiz_App.git
   cd backend
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv myenv
   
   # On Windows
   myenv\Scripts\activate
   
   # On macOS/Linux
   source myenv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

5. **Run the server**
   ```bash
   python app.py
   ```

   The server will start at `http://localhost:5000`

## 📚 API Endpoints

### Upload PDF & Generate Quiz
```
POST /upload
```
- **Description**: Upload a PDF file to generate quiz questions
- **Content-Type**: multipart/form-data
- **Parameters**:
  - `file` (required): PDF file
- **Response**:
  ```json
  {
    "quizId": "507f1f77bcf86cd799439011",
    "quizQuestions": [
      {
        "question": "What is...?",
        "choices": ["Option A", "Option B", "Option C", "Option D"],
        "correctAnswer": "Option A",
        "explanation": "Explanation text"
      }
    ]
  }
  ```

### Submit Quiz Result
```
POST /submit-result
```
- **Description**: Submit user's quiz answers and score
- **Content-Type**: application/json
- **Body**:
  ```json
  {
    "quizId": "507f1f77bcf86cd799439011",
    "selectedAnswers": ["Option A", "Option B"],
    "score": 8,
    "totalQuestions": 10,
    "timeSpent": 300
  }
  ```
- **Response**:
  ```json
  {
    "message": "Result submitted",
    "resultId": "507f1f77bcf86cd799439012"
  }
  ```

### Get Quiz Statistics
```
GET /quiz-stats/<quiz_id>
```
- **Description**: Get statistics for a specific quiz
- **Response**:
  ```json
  {
    "times_taken": 5,
    "average_score": 75.5,
    "highest_score": 95.0
  }
  ```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `MONGODB_URI` | MongoDB connection string | `mongodb+srv://...` |
| `GEMINI_API_KEY` | Google Gemini API key | `AIzaSy...` |

### CORS Settings

Currently configured for local development with React at `http://localhost:3000`. Modify in `app.py` if needed:

```python
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
```

## 📁 Project Structure

```
backend/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
├── myenv/                # Virtual environment
└── README.md             # This file
```

## 🧪 Quiz Generation Process

1. User uploads a PDF file
2. Text is extracted from the PDF using PyMuPDF
3. Text is sent to Google Gemini API with a structured prompt
4. API generates multiple-choice questions with explanations
5. Questions are parsed and stored in MongoDB
6. Quiz ID is returned to the user

### Fallback Mechanism

If the Gemini API fails or returns empty:
- System extracts key sentences from the text
- Generates basic quiz questions from these sentences
- Ensures users always get some quiz content

## 🔐 Security Notes

- Store `.env` file securely and **never commit it to version control**
- Use environment variables for all sensitive data
- MongoDB connection should use authentication
- Consider adding rate limiting for production
- Add input validation for production use

## 📦 Dependencies

Key packages included:
- `Flask` - Web framework
- `pymongo` - MongoDB driver
- `google-generativeai` - Google Gemini API
- `PyMuPDF` - PDF text extraction
- `flask-cors` - CORS support
- `python-dotenv` - Environment variable management
- `gunicorn` - Production server
- `pydantic` - Data validation

## 🚀 Deployment

For production deployment:

1. Set `debug=False` in the Flask app
2. Use Gunicorn:
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```
3. Use a reverse proxy like Nginx
4. Deploy to cloud platforms (Heroku, Railway, Pythonanywhere, etc.)

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `GEMINI_API_KEY not found` | Check `.env` file and ensure the key is set |
| `MongoDB connection error` | Verify `MONGODB_URI` and network access |
| `CORS error` | Update CORS origin to match your frontend URL |
| `Empty quiz generated` | Try a PDF with more text content |

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Support

For issues, questions, or suggestions, please open an issue on the repository.

---

**Made with ❤️ for educational purposes**
