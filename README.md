# Video Games Q&A Chatbot

A locally-hosted, full-stack web application featuring a custom fine-tuned Qwen 2B Large Language Model. This chatbot is specifically trained to retrieve release dates and critic scores for over 64,000 video games.

The project utilizes a React/Vite frontend with a dark-mode user interface and a FastAPI backend configured to run inference on CPU with optimized memory usage (bfloat16) and deterministic generation (greedy decoding).

---

## Project Architecture

*   **Backend:** Python, FastAPI, PyTorch, Hugging Face Transformers.
*   **Frontend:** JavaScript, React, Vite, CSS.
*   **Model:** Fine-tuned `Qwen3.5-2B-Base`.
*   **Dataset:** 64,000+ video game records detailing titles, release dates, and critic ratings across multiple consoles.

---

## Prerequisites

Before running this project, ensure you have the following installed on your machine:
*   **Python 3.8+**
*   **Node.js (v16+ recommended) and npm**
*   **Git**

---

## Installation & Setup

Follow these steps to get both the backend and frontend servers running on your local machine.

### 1. Clone the Repository

Open your terminal and clone the project to your local machine:

```powershell
git clone https://github.com/mgasserr/video-game-qa-chatbot.git
cd video-game-qa-chatbot
```

### 2. Backend Setup (FastAPI & LLM)

Open a terminal in the root directory of the project and run the following commands to set up the Python environment.

**Create and activate a virtual environment (Windows):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Install backend dependencies:**
```powershell
cd backend
pip install -r requirements.txt
```

**Start the FastAPI server:**
```powershell
uvicorn main:app --reload --port 8000
```
> Note: The backend server must remain running in this terminal window for the chatbot to generate answers. The API will be available at `http://localhost:8000`.

### 3. Frontend Setup (React & Vite)

Open a **new** terminal window (leave the backend terminal running), navigate to the root of the project, and run the following commands.

**Navigate to the frontend directory and install dependencies:**
```powershell
cd frontend
npm install
```

**Start the React development server:**
```powershell
npm run dev
```
> Note: After running this command, Vite will provide a local URL (usually `http://localhost:5173`). Open this URL in your web browser to interact with the chatbot.

---

## Usage Guidelines: How to Ask Questions

Because the underlying LLM was fine-tuned programmatically on strict text templates, it requires exact sentence structures to return accurate data. Deviating from these templates may result in hallucinations.

Ensure you spell the game title exactly as it appears in the database (e.g., "Grand Theft Auto V", not "GTA 5").

### Supported Query Types

**1. Release Date Queries**
You must use the exact phrasing:
*   `When did the game [Game Title] come out?`
*   *Example:* `When did the game Minecraft come out?`

**2. Critic Rating Queries**
You must use the exact phrasing:
*   `What are the reviews and ratings for [Game Title]?`
*   *Example:* `What are the reviews and ratings for Portal 2?`

---

## Current Limitations & Future Roadmap

*   **Platform Duplication:** Because the training template did not specify the console (e.g., PS3 vs. Xbox 360), games released on multiple platforms may result in conflicting data predictions from the model. 
*   **Future Implementation (RAG):** The next phase of this project involves migrating from direct-weight memorization to a Retrieval-Augmented Generation (RAG) architecture. This will bypass the limitations of LLM floating-point tokenization, ensuring 100% factual accuracy by directly querying the dataset at runtime.