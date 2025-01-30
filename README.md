# AI-Powered Quiz Generator - Backend

## **Overview**
The **AI-Powered Quiz Generator** backend is built using **Flask** and handles PDF text extraction, AI-based quiz generation, result storage, and analytics. It integrates with **Google's Gemini API** to generate multiple-choice questions and uses **MongoDB** to store quiz data and results.

## **Features**
- **PDF Text Extraction** – Extracts text from uploaded PDFs using **PyMuPDF (fitz)**.
- **AI-Powered Quiz Generation** – Uses **Google's Gemini API** to create quizzes from extracted text.
- **Result Processing & Analytics** – Calculates scores and provides statistics such as average score, highest/lowest scores, and total attempts.
- **REST API Integration** – Provides endpoints for frontend communication.
- **Deployed on Render** – Ensures scalability and easy access.

## **Tech Stack**
- **Flask** – Lightweight Python framework for API handling.
- **PyMuPDF (fitz)** – Extracts text from PDF files.
- **Google Gemini API** – AI-based question generation.
- **MongoDB** – Stores quizzes, user results, and analytics.
- **Render** – Deployment platform for backend services.

## **Installation & Setup**
1. **Clone the Repository:**
   ```sh
   git clone https://github.com/karthikeyamaddu/Quiz_backend.git
   cd Quiz_backend
   ```
2. **Create a Virtual Environment (Optional but Recommended):**
   ```sh
   python -m venv venv
   source venv/bin/activate   # On macOS/Linux
   venv\Scripts\activate      # On Windows
   ```
3. **Install Dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Set Up Environment Variables:**
   - Create a `.env` file and add your **Google Gemini API Key** and **MongoDB connection string**.
   ```
   GEMINI_API_KEY=your_api_key
   MONGO_URI=your_mongodb_uri
   ```
5. **Run the Flask Server:**
   ```sh
   flask run
   ```
6. **Test API Locally:**
   - Open `http://127.0.0.1:5000/` in your browser or use Postman to test API endpoints.

## **API Endpoints**
| Endpoint               | Method | Description |
|------------------------|--------|-------------|
| `/upload`             | POST   | Uploads a PDF, extracts text, and generates a quiz. |
| `/submit-result`      | POST   | Submits quiz answers and calculates scores. |
| `/quiz-stats/<quiz_id>` | GET    | Retrieves statistics for a specific quiz. |

## **Deployment**
The backend is deployed on **Render** and can be accessed at:
[Quiz Backend](https://quiz-app-d38s.onrender.com)

## **Future Enhancements**
- **User Authentication** – Enable login to track individual quiz attempts.
- **Quiz Customization** – Allow users to select question count and difficulty level.
- **Support for More File Types** – Extend support to **Word and plain text files**.
- **Leaderboard** – Display top performers.
- **Advanced Analytics** – Track time per question and performance trends.

## **Contributing**
Contributions are welcome! Fork the repo, create a new branch, and submit a pull request.

## **License**
This project is licensed under the **MIT License**.

