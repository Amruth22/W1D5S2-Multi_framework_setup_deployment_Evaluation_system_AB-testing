# Complete AI Project with LangChain, Gemini, Flask, and Evaluation

This project demonstrates a comprehensive AI backend system using Flask, LangChain, and multiple Gemini models, including framework implementation, multi-model setup, evaluation, and A/B testing, all within a flat file structure.

## Features:
- **Flask API:** RESTful endpoints for AI functionalities.
- **LangChain Integration:** Orchestrates interactions with Gemini models.
- **Multiple Gemini Models:** Uses `gemini-1.5-pro`, `gemini-1.5-flash`, and `gemini-pro-vision`.
- **Framework Implementation:** Examples of LangChain-based AI capabilities (summarization, Q&A, image analysis).
- **Multi-model Setup:** Dynamic selection and use of different Gemini models based on task.
- **Model Deployment:** AI capabilities exposed via API.
- **Evaluation Pipeline:** System for comparing different model outputs.
- **Evaluation System:** Logging and analysis of model performance.
- **A/B Testing:** Logic for managing and analyzing A/B experiments.
- **Swagger UI:** Interactive API documentation.

## Setup Instructions:

1.  **Clone the repository (if applicable).**

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure your Gemini API Key:**
    Open the `.env` file (or create it if it doesn't exist) and add your Gemini API key:
    ```
    GEMINI_API_KEY="YOUR_ACTUAL_GEMINI_API_KEY"
    ```
    (Replace `"YOUR_ACTUAL_GEMINI_API_KEY"` with your key).

## How to Run:

### 1. Run the Flask API:
```bash
python app.py
```
The API will typically run on `http://127.0.0.1:5000/`.

### 2. Access Swagger UI:
Once the Flask app is running, open your web browser and navigate to:
`http://127.0.0.1:5000/apidocs`
Here you can explore and test the API endpoints.

### 3. Run Evaluation/A/B Testing Scripts:
(Instructions for running specific evaluation or A/B testing scenarios will be provided within `evaluation_and_ab_testing.py`'s `if __name__ == "__main__":` block.)
