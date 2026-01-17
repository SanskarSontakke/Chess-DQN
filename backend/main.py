from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import chess
import torch
import random
import time
from model import ChessDQN, board_to_tensor, fen_to_tensor_fast
from chess_env import ChessEnv
from trainer import Trainer, MODEL_PATH
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Import strategic search engine for lookahead
from strategic_search import search_engine, get_best_move_with_thinking
from strategic_eval import get_full_strategic_evaluation, get_game_phase

# Import hybrid search for DQN + Minimax combination
from hybrid_search import HybridSearchEngine

# Import logging system
from logger import (
    logger, log_system_info, log_training_start, log_simulation_progress,
    log_simulation_complete, log_training_progress, log_training_complete,
    log_model_saved, log_model_loaded, log_move_evaluation, log_session_summary,
    log_error, log_warning
)

# Dashboard path
DASHBOARD_PATH = os.path.join(os.path.dirname(__file__), "dashboard.html")

# Global training control flags
stop_training_requested = False
training_mode = "random"  # Options: "random", "model", "hybrid"

# ========================================================================
# APP SETUP
# ========================================================================
app = FastAPI(title="Chess DQN API", description="Chess AI Training Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================================================
# GLOBAL MODELS & DEVICE
# ========================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Log system info on startup
log_system_info()

model = ChessDQN().to(device)
target_model = ChessDQN().to(device)
target_model.load_state_dict(model.state_dict())
trainer = Trainer(model, target_model, device=device)

# Load model from disk if exists
if trainer.load_model():
    log_model_loaded(MODEL_PATH)

env = ChessEnv()

# Hybrid search engine (combines Minimax with DQN evaluation)
hybrid_engine = HybridSearchEngine(model, device)

# ========================================================================
# REQUEST MODELS
# ========================================================================
class MoveRequest(BaseModel):
    fen: str

class TrainRequest(BaseModel):
    games: int = 100
    mode: str = None  # Optional: "random", "model", "hybrid"

# ========================================================================
# ENDPOINTS
# ========================================================================
@app.get("/")
def read_root():
    return {
        "status": "Chess AI Backend Running",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }

@app.post("/move")
def get_move(request: MoveRequest):
    start_time = time.time()
    board = chess.Board(request.fen)
    legal_moves = list(board.legal_moves)
    
    if not legal_moves:
        log_warning(f"No legal moves for position: {request.fen[:30]}...")
        return {"error": "No legal moves available"}

    # Batch evaluation for GPU efficiency
    next_states = []
    for move in legal_moves:
        board.push(move)
        next_states.append(board_to_tensor(board))
        board.pop()
    
    # Stack all tensors: List[1, 14, 8, 8] -> [N, 14, 8, 8]
    batch_tensor = torch.cat(next_states, dim=0).to(device)
    
    with torch.no_grad():
        # returns [N, 1] -> [N]
        q_values = model(batch_tensor).squeeze(1).tolist()
    
    evaluations = []
    max_q = -float('inf')
    best_move = None
    
    for i, move in enumerate(legal_moves):
        q = q_values[i]
        evaluations.append({
            "move": move.uci(),
            "q_value": q
        })
        if q > max_q:
            max_q = q
            best_move = move

    # Sort evaluations for thought process visualization
    evaluations = sorted(evaluations, key=lambda x: x["q_value"], reverse=True)
    
    # Log move evaluation
    elapsed_ms = (time.time() - start_time) * 1000
    log_move_evaluation(
        fen_short=request.fen[:20],
        num_moves=len(legal_moves),
        best_move=best_move.uci(),
        q_value=max_q,
        time_ms=elapsed_ms
    )

    return {
        "best_move": best_move.uci(),
        "q_value": max_q,
        "thought_process": evaluations[:5]  # Top 5 moves
    }


class StrategicMoveRequest(BaseModel):
    fen: str
    depth: int = 3  # Search depth (1-5 recommended)


@app.post("/strategic-move")
def get_strategic_move(request: StrategicMoveRequest):
    """
    Get AI move using Minimax search with strategic evaluation.
    Returns move with detailed thinking process and strategy logging.
    """
    start_time = time.time()
    
    try:
        board = chess.Board(request.fen)
    except ValueError as e:
        return {"error": f"Invalid FEN: {e}"}
    
    if board.is_game_over():
        return {"error": "Game is already over"}
    
    # Clamp depth between 1 and 5
    depth = max(1, min(5, request.depth))
    
    try:
        best_move, thinking_log = search_engine.search(board, depth=depth)
        formatted_log = search_engine.format_thinking_log(thinking_log)
        
        # Get candidate moves for frontend display
        candidates = []
        for move_eval in thinking_log.candidate_moves[:5]:
            candidates.append({
                "move": move_eval.move.uci(),
                "san": board.san(move_eval.move),
                "score": round(move_eval.score, 3),
                "reasoning": move_eval.strategic_reasoning
            })
        
        # Principal variation in SAN notation
        pv_san = []
        temp_board = board.copy()
        for move in thinking_log.principal_variation[:6]:
            pv_san.append(temp_board.san(move))
            temp_board.push(move)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Strategic move: {best_move.uci()} | Depth: {depth} | Score: {thinking_log.best_score:.2f} | Time: {elapsed_ms:.1f}ms")
        
        return {
            "best_move": best_move.uci(),
            "best_move_san": board.san(best_move),
            "score": round(thinking_log.best_score, 3),
            "depth": depth,
            "game_phase": thinking_log.game_phase,
            "time_ms": round(elapsed_ms, 1),
            "candidates": candidates,
            "principal_variation": pv_san,
            "strategic_factors": thinking_log.strategic_assessment,
            "thinking_log": formatted_log
        }
    
    except Exception as e:
        logger.error(f"Strategic move error: {e}")
        return {"error": str(e)}


@app.get("/ai-thinking")
def get_ai_thinking():
    """Get recent AI thinking logs."""
    logs = search_engine.get_recent_logs(n=5)
    formatted_logs = [search_engine.format_thinking_log(log) for log in logs]
    return {
        "total_logs": len(search_engine.thinking_logs),
        "recent_logs": formatted_logs
    }


@app.post("/clear-thinking-logs")
def clear_thinking_logs():
    """Clear AI thinking logs."""
    search_engine.clear_logs()
    hybrid_engine.clear_logs()
    return {"message": "Thinking logs cleared"}


class HybridMoveRequest(BaseModel):
    fen: str
    depth: int = 3  # Search depth (1-5 recommended)


@app.post("/hybrid-move")
def get_hybrid_move(request: HybridMoveRequest):
    """
    Get AI move using HYBRID DQN-Minimax search.
    
    This combines:
    - Minimax lookahead (sees N moves ahead)
    - Neural network evaluation (learned from training)
    
    The DQN evaluates leaf positions instead of handcrafted heuristics.
    This allows the trained model to plan strategically!
    """
    start_time = time.time()
    
    try:
        board = chess.Board(request.fen)
    except ValueError as e:
        return {"error": f"Invalid FEN: {e}"}
    
    if board.is_game_over():
        return {"error": "Game is already over"}
    
    # Clamp depth between 1 and 5
    depth = max(1, min(5, request.depth))
    
    try:
        best_move, thinking_log = hybrid_engine.search(board, depth=depth)
        formatted_log = hybrid_engine.format_thinking_log(thinking_log)
        
        # Get candidate moves for frontend display
        candidates = []
        for move_eval in thinking_log.candidate_moves[:5]:
            candidates.append({
                "move": move_eval.move.uci(),
                "san": board.san(move_eval.move),
                "score": round(move_eval.score, 3),
                "reasoning": move_eval.reasoning
            })
        
        # Principal variation in SAN notation
        pv_san = []
        temp_board = board.copy()
        for move in thinking_log.principal_variation[:6]:
            pv_san.append(temp_board.san(move))
            temp_board.push(move)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        logger.info(
            f"Hybrid move: {best_move.uci()} | Depth: {depth} | "
            f"Score: {thinking_log.best_score:.2f} | "
            f"Nodes: {thinking_log.nodes_evaluated} | "
            f"NN Evals: {thinking_log.model_evaluations} | "
            f"Time: {elapsed_ms:.1f}ms"
        )
        
        return {
            "best_move": best_move.uci(),
            "best_move_san": board.san(best_move),
            "score": round(thinking_log.best_score, 3),
            "depth": depth,
            "game_phase": thinking_log.game_phase,
            "time_ms": round(elapsed_ms, 1),
            "nodes_evaluated": thinking_log.nodes_evaluated,
            "model_evaluations": thinking_log.model_evaluations,
            "candidates": candidates,
            "principal_variation": pv_san,
            "strategic_factors": thinking_log.strategic_assessment,
            "thinking_log": formatted_log,
            "mode": "hybrid_dqn_minimax"
        }
    
    except Exception as e:
        logger.error(f"Hybrid move error: {e}")
        return {"error": str(e)}

# ========================================================================
# TRAINING STATE
# ========================================================================
training_progress = {
    "current": 0,
    "total": 0,
    "status": "idle",  # idle, simulating, training, completed
    "games_per_sec": 0.0,
    "phase": "",
    "elapsed_time": 0.0,
    # Training phase specific
    "training_current": 0,
    "training_total": 0,
    "training_speed": 0.0,  # updates/sec
    "current_loss": 0.0
}

# ========================================================================
# PARALLEL GAME SIMULATION (CPU-bound, use multiprocessing)
# ========================================================================
def simulate_single_game(_):
    """
    Simulates one full game and returns lightweight experience tuples.
    Uses FEN strings instead of numpy arrays for ~3x faster collection.
    Tensor conversion happens in batch after all games complete.
    """
    board = chess.Board()
    experiences = []
    done = False
    
    while not done:
        state_fen = board.fen()
        legal_moves = list(board.legal_moves)
        
        # Random policy for simulation (faster than model inference)
        move = random.choice(legal_moves)
        
        # Execute move
        board.push(move)
        
        # Check game over
        done = board.is_game_over()
        reward = 0.0
        
        if done:
            result = board.result()
            if result == "1-0":
                reward = 10.0
            elif result == "0-1":
                reward = -10.0
            else:
                reward = 0.5
        
        next_fen = board.fen()
        experiences.append((state_fen, reward, next_fen, done))
    
    return experiences


def run_training_optimized(num_games: int):
    """
    Optimized training with comprehensive logging:
    1. Simulate N games using the selected training mode.
    2. Collect all experiences.
    3. Batch train on GPU with large batch sizes.
    4. Save model to disk.
    """
    global training_progress, stop_training_requested, training_mode
    
    # Initialize - RESET ALL progress fields for new session
    stop_training_requested = False
    training_progress["current"] = 0
    training_progress["total"] = num_games
    training_progress["status"] = "simulating"
    training_progress["phase"] = f"Game Simulation ({training_mode.upper()})"
    training_progress["elapsed_time"] = 0.0
    # Reset training phase fields too
    training_progress["training_current"] = 0
    training_progress["training_total"] = 0
    training_progress["training_speed"] = 0.0
    training_progress["current_loss"] = 0.0

    
    num_workers = min(multiprocessing.cpu_count(), 12)
    
    # Log training start
    log_training_start(
        num_games=num_games,
        batch_size=trainer.batch_size,
        device=str(device)
    )
    logger.info(f"Training Mode: {training_mode.upper()}")
    logger.info(f"Using {num_workers} CPU workers for parallel simulation")
    
    start_time = time.time()
    final_loss = 0.0
    
    # ====================================================================
    # PHASE 1: GAME SIMULATION (mode-dependent)
    # ====================================================================
    games_done = 0
    
    if training_mode == "random":
        # RANDOM MODE: Fast parallel processing (CPU workers)
        batch_sim_size = min(num_workers * 4, num_games)
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            while games_done < num_games:
                if stop_training_requested:
                    logger.warning("Stop training requested during simulation phase")
                    break

                batch_count = min(batch_sim_size, num_games - games_done)
                futures = [executor.submit(simulate_single_game, i) for i in range(batch_count)]
                
                for future in as_completed(futures):
                    if stop_training_requested:
                        break
                    
                    exps = future.result()
                    for (state_fen, r, next_fen, d) in exps:
                        state_tensor = torch.from_numpy(fen_to_tensor_fast(state_fen)).unsqueeze(0)
                        next_tensor = torch.from_numpy(fen_to_tensor_fast(next_fen)).unsqueeze(0)
                        trainer.memory.push(state_tensor, r, next_tensor, d)
                    games_done += 1
                    training_progress["current"] = games_done
                    
                    if games_done % max(1, num_games // 10) == 0:
                        elapsed = time.time() - start_time
                        gps = games_done / elapsed if elapsed > 0 else 0
                        training_progress["games_per_sec"] = gps
                        training_progress["elapsed_time"] = elapsed
                        log_simulation_progress(games_done, num_games, elapsed, gps)
    
    else:
        # MODEL or HYBRID MODE: Sequential processing (uses GPU model)
        from smart_simulation import simulate_game_model_guided, simulate_game_hybrid
        
        for game_idx in range(num_games):
            if stop_training_requested:
                logger.warning("Stop training requested during simulation phase")
                break
            
            # Select simulation function based on mode
            if training_mode == "model":
                exps = simulate_game_model_guided(model, device, epsilon=0.1)
            else:  # hybrid
                exps = simulate_game_hybrid(model, device, depth=2, epsilon=0.05)
            
            # Store experiences
            for (state_fen, r, next_fen, d) in exps:
                state_tensor = torch.from_numpy(fen_to_tensor_fast(state_fen)).unsqueeze(0)
                next_tensor = torch.from_numpy(fen_to_tensor_fast(next_fen)).unsqueeze(0)
                trainer.memory.push(state_tensor, r, next_tensor, d)
            
            games_done += 1
            training_progress["current"] = games_done
            
            if games_done % max(1, num_games // 20) == 0:
                elapsed = time.time() - start_time
                gps = games_done / elapsed if elapsed > 0 else 0
                training_progress["games_per_sec"] = gps
                training_progress["elapsed_time"] = elapsed
                log_simulation_progress(games_done, num_games, elapsed, gps)

    sim_time = time.time() - start_time
    log_simulation_complete(games_done, sim_time, len(trainer.memory))
    
    if stop_training_requested:
        # Final summary - update status to idle
        training_progress["status"] = "idle"
        training_progress["phase"] = "Stopped by User"
        logger.info("Training session STOPPED during simulation")
        return

    # ====================================================================
    # PHASE 2: GPU TRAINING
    # ====================================================================
    training_progress["status"] = "training"
    training_progress["phase"] = "Neural Network Training"
    
    logger.info(f"Starting GPU training on {len(trainer.memory):,} experiences")
    logger.info(f"Batch Size: {trainer.batch_size} | Device: {device}")
    
    train_start = time.time()
    num_train_updates = max(1, len(trainer.memory) // trainer.batch_size)
    updates_per_step = 10
    
    # Initialize training progress tracking
    training_progress["training_total"] = num_train_updates
    training_progress["training_current"] = 0
    
    for i in range(0, num_train_updates, updates_per_step):
        # Check for stop request
        if stop_training_requested:
            logger.warning("Stop training requested during training phase")
            break

        step_start = time.time()
        loss = trainer.train_step(num_updates=updates_per_step)
        step_time = time.time() - step_start
        
        final_loss = loss if loss else 0.0
        
        # Update training progress
        training_progress["training_current"] = min(i + updates_per_step, num_train_updates)
        training_progress["current_loss"] = final_loss
        training_progress["training_speed"] = updates_per_step / step_time if step_time > 0 else 0
        training_progress["elapsed_time"] = time.time() - start_time
        
        if (i // updates_per_step) % 5 == 0:
            trainer.update_target_network()
        
        # Log every 20%
        if i % max(1, num_train_updates // 5) == 0:
            elapsed = time.time() - train_start
            log_training_progress(min(i + updates_per_step, num_train_updates), num_train_updates, final_loss, elapsed)
    
    train_time = time.time() - train_start
    log_training_complete(min(training_progress["training_current"], num_train_updates), train_time, final_loss)
    
    if stop_training_requested:
        # Final summary - update status to idle
        training_progress["status"] = "idle"
        training_progress["phase"] = "Stopped by User"
        logger.info("Training session STOPPED during GPU training")
        return

    
    # ====================================================================
    # PHASE 3: SAVE MODEL & UPDATE STATS
    # ====================================================================
    trainer.save_model()
    model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # MB
    log_model_saved(MODEL_PATH, model_size)
    
    # Update model statistics
    model_stats["total_games"] += num_games
    model_stats["training_sessions"] += 1
    model_stats["experiences"] = len(trainer.memory)
    
    # Calculate total time BEFORE using it
    total_time = time.time() - start_time
    
    model_stats["history"].append({
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "type": "training",
        "title": f"Training Complete: {num_games:,} Games",
        "details": f"Duration: {total_time:.1f}s | Speed: {num_games/sim_time:.1f} g/s | Loss: {final_loss:.6f} | Size: {model_size:.2f} MB"
    })
    
    # Final summary - update status to completed
    training_progress["status"] = "completed"
    training_progress["phase"] = "Complete"
    training_progress["elapsed_time"] = total_time
    training_progress["current_loss"] = final_loss
    
    log_session_summary(total_time, num_games, final_loss)
    logger.info("Training status updated to COMPLETED")



@app.post("/train")
def train_ai(request: TrainRequest, background_tasks: BackgroundTasks):
    global training_mode
    if request.mode:
        training_mode = request.mode
    logger.info(f"API: Training request received for {request.games:,} games (mode: {training_mode})")
    background_tasks.add_task(run_training_optimized, request.games)
    return {"message": f"Training session started for {request.games} games (mode: {training_mode})"}


@app.post("/set-training-mode")
def set_training_mode(mode: str):
    """Set the training mode: random, model, or hybrid."""
    global training_mode
    if mode not in ["random", "model", "hybrid"]:
        return {"error": f"Invalid mode: {mode}. Use 'random', 'model', or 'hybrid'."}
    training_mode = mode
    logger.info(f"Training mode set to: {training_mode}")
    return {"message": f"Training mode set to: {training_mode}", "mode": training_mode}


@app.get("/training-mode")
def get_training_mode():
    """Get the current training mode."""
    return {"mode": training_mode}

@app.post("/stop-training")
def stop_training():
    global stop_training_requested
    stop_training_requested = True
    logger.info("API: Stop training request received")
    return {"message": "Stop training request received"}

@app.get("/training-status")
def get_training_status():
    return training_progress

@app.post("/save-model")
def save_model_endpoint():
    trainer.save_model()
    model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    log_model_saved(MODEL_PATH, model_size)
    return {"message": "Model saved successfully"}

@app.post("/load-model")
def load_model_endpoint():
    success = trainer.load_model()
    if success:
        log_model_loaded(MODEL_PATH)
    return {"message": "Model loaded successfully" if success else "No saved model found"}

@app.get("/system-info")
def get_system_info():
    """Return detailed system information."""
    info = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "pytorch_version": torch.__version__,
        "cpu_count": multiprocessing.cpu_count(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        info["cuda_version"] = torch.version.cuda
    return info

@app.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard():
    """Serve the training dashboard HTML page."""
    if os.path.exists(DASHBOARD_PATH):
        with open(DASHBOARD_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Dashboard not found</h1>", status_code=404)


# ========================================================================
# MODEL INFO & HYPERPARAMETERS
# ========================================================================
class HyperparamsRequest(BaseModel):
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon: float = 0.1


# Model statistics tracking
model_stats = {
    "total_games": 0,
    "training_sessions": 0,
    "experiences": 0,
    "history": []
}


@app.get("/model-info")
def get_model_info():
    """Return model information including parameters and training stats."""
    try:
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
    except Exception as e:
        logger.error(f"Error counting parameters: {e}")
        total_params = 0
    
    # Model size
    model_size_mb = 0.0
    try:
        if os.path.exists(MODEL_PATH):
            model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    except Exception as e:
        logger.error(f"Error getting model size: {e}")
    
    # Get trainer attributes safely
    lr = getattr(trainer, 'lr', 0.001)
    gamma = getattr(trainer, 'gamma', 0.99)
    epsilon = getattr(trainer, 'epsilon', 0.1)
    experiences = len(trainer.memory) if hasattr(trainer, 'memory') else 0
    
    return {
        "parameters": total_params,
        "size_mb": round(model_size_mb, 2),
        "total_games": model_stats["total_games"],
        "training_sessions": model_stats["training_sessions"],
        "experiences": experiences,
        "learning_rate": lr,
        "gamma": gamma,
        "epsilon": epsilon
    }


@app.post("/set-hyperparams")
def set_hyperparams(request: HyperparamsRequest):
    """Update training hyperparameters."""
    trainer.lr = request.learning_rate
    trainer.gamma = request.gamma
    trainer.epsilon = request.epsilon
    
    # Update optimizer learning rate
    for param_group in trainer.optimizer.param_groups:
        param_group['lr'] = request.learning_rate
    
    logger.info(f"Hyperparameters updated: LR={request.learning_rate}, γ={request.gamma}, ε={request.epsilon}")
    
    # Add to history
    model_stats["history"].append({
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "type": "config",
        "title": "Hyperparameters Updated",
        "details": f"LR: {request.learning_rate}, γ: {request.gamma}, ε: {request.epsilon}"
    })
    
    return {"message": "Hyperparameters updated successfully"}


@app.post("/reset-model")
def reset_model():
    """Reset the model to random weights and clear all training history."""
    global model, target_model, trainer, model_stats
    
    # Delete saved model file
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
        logger.info(f"Deleted model file: {MODEL_PATH}")
    
    # Re-initialize models
    model = ChessDQN().to(device)
    target_model = ChessDQN().to(device)
    target_model.load_state_dict(model.state_dict())
    trainer = Trainer(model, target_model, device=device)
    
    # Reset stats
    model_stats = {
        "total_games": 0,
        "training_sessions": 0,
        "experiences": 0,
        "history": [{
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "type": "reset",
            "title": "Model Reset",
            "details": "All weights cleared and re-initialized to random values"
        }]
    }
    
    logger.warning("Model has been reset to random weights!")
    return {"message": "Model reset successfully"}


@app.get("/model-history")
def get_model_history():
    """Return training history timeline."""
    return {"history": model_stats["history"][-50:]}  # Last 50 events
