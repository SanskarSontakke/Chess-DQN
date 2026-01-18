# ♟️ Chess DQN AI

A Deep Q-Network (DQN) based Chess AI with a powerful dashboard for real-time training visualization and analysis.

## 🚀 Features

*   **Interactive Dashboard**: Visualize training progress, loss, and win rates in real-time.
*   **Three Training Modes**:
    *   **Random**: Fast generation of games for initial exploration.
    *   **Model-Guided**: Uses the neural network with epsilon-greedy exploration.
    *   **Hybrid**: Combines Minimax lookahead with DQN evaluation (Grandmaster mode).
*   **Live Analysis**: See what the AI is "thinking" with real-time arrow visualizers and score outputs.
*   **Cloud Deployment**: Ready logic for remote GPU training (Google Colab / Kaggle).

## 🛠️ Installation

### Backend (Python/FastAPI)

1.  Navigate to the backend directory:
    ```bash
    cd backend
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Start the server:
    ```bash
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```

### Frontend (Angular)

1.  Navigate to the frontend directory:
    ```bash
    cd frontend
    ```
2.  Install packages:
    ```bash
    npm install
    ```
3.  Run the application:
    ```bash
    npm start
    ```
4.  Open `http://localhost:4200` in your browser.

## 🧠 Training Guide

### 1. Random Mode (Speed: ~100 games/sec)
*   **Use case**: Best for filling the Experience Replay buffer quickly when starting from scratch.
*   **Description**: plays random legal moves to generate raw data.

### 2. Model-Guided Mode (Speed: ~20 games/sec)
*   **Use case**: Standard training phase.
*   **Description**: The AI uses its current neural network to play, helping it reinforce good strategies it has discovered.

### 3. Hybrid Mode (Speed: ~2-5 games/sec)
*   **Use case**: Fine-tuning and high-quality data generation.
*   **Description**: Performs a 2-ply Minimax search for every move, using the DQN to evaluate leaf nodes. Extremely high-quality play but computationally expensive.

## 📂 Project Structure

*   `backend/`: FastAPI server, PyTorch model, and reinforcement learning logic.
    *   `main.py`: API endpoints and WebSocket handlers.
    *   `model.py`: The CNN-based Deep Q-Network architecture.
    *   `trainer.py`: Training loop and Experience Replay buffer.
    *   `smart_simulation.py`: Logic for different simulation modes (Random/Hybrid).
*   `frontend/`: Angular-based UI for the dashboard and chess board.
