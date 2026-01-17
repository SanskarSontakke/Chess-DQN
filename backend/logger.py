"""
Chess DQN Logging System
Comprehensive logging with performance metrics, device info, and training details.
"""
import logging
import sys
import os
from datetime import datetime
from typing import Optional
import torch

# Create logs directory
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Log file path with timestamp
LOG_FILE = os.path.join(LOG_DIR, f"chess_dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Add color to level name
        record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)


def setup_logger(name: str = "ChessDQN") -> logging.Logger:
    """Set up and return a configured logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = ColoredFormatter(
        fmt='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    # File handler with full details
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


# Global logger instance
logger = setup_logger()


def log_system_info():
    """Log system and device information."""
    logger.info("=" * 60)
    logger.info("CHESS DQN SYSTEM STARTUP")
    logger.info("=" * 60)
    
    # Python info
    logger.info(f"Python Version: {sys.version}")
    
    # PyTorch info
    logger.info(f"PyTorch Version: {torch.__version__}")
    
    # CUDA info
    if torch.cuda.is_available():
        logger.info(f"CUDA Available: ✓ YES")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        logger.info(f"GPU Compute Capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")
    else:
        logger.warning(f"CUDA Available: ✗ NO (Using CPU)")
    
    # Log file location
    logger.info(f"Log File: {LOG_FILE}")
    logger.info("=" * 60)


def log_training_start(num_games: int, batch_size: int, device: str):
    """Log training session start."""
    logger.info("=" * 60)
    logger.info("TRAINING SESSION STARTED")
    logger.info("=" * 60)
    logger.info(f"Number of Games: {num_games:,}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Device: {device}")
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def log_simulation_progress(current: int, total: int, elapsed_time: float, games_per_sec: float):
    """Log simulation progress."""
    percent = (current / total) * 100
    logger.info(f"Simulation: {current:,}/{total:,} games ({percent:.1f}%) | Speed: {games_per_sec:.1f} games/sec | Elapsed: {elapsed_time:.1f}s")


def log_simulation_complete(total_games: int, total_time: float, experiences: int):
    """Log simulation completion."""
    speed = total_games / total_time if total_time > 0 else 0
    logger.info("-" * 60)
    logger.info("SIMULATION COMPLETE")
    logger.info(f"Total Games: {total_games:,}")
    logger.info(f"Total Time: {total_time:.2f}s")
    logger.info(f"Average Speed: {speed:.1f} games/sec")
    logger.info(f"Experiences Collected: {experiences:,}")
    logger.info("-" * 60)


def log_training_progress(update: int, total_updates: int, loss: float, elapsed_time: float):
    """Log training progress."""
    percent = (update / total_updates) * 100
    updates_per_sec = update / elapsed_time if elapsed_time > 0 else 0
    logger.info(f"Training: {update:,}/{total_updates:,} updates ({percent:.1f}%) | Loss: {loss:.6f} | Speed: {updates_per_sec:.1f} updates/sec")


def log_training_complete(total_updates: int, total_time: float, final_loss: float):
    """Log training completion."""
    speed = total_updates / total_time if total_time > 0 else 0
    logger.info("-" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Total Updates: {total_updates:,}")
    logger.info(f"Total Time: {total_time:.2f}s")
    logger.info(f"Average Speed: {speed:.1f} updates/sec")
    logger.info(f"Final Loss: {final_loss:.6f}")
    logger.info("-" * 60)


def log_model_saved(path: str, model_size_mb: Optional[float] = None):
    """Log model save event."""
    if model_size_mb:
        logger.info(f"Model Saved: {path} ({model_size_mb:.2f} MB)")
    else:
        logger.info(f"Model Saved: {path}")


def log_model_loaded(path: str):
    """Log model load event."""
    logger.info(f"Model Loaded: {path}")


def log_api_request(endpoint: str, method: str, details: str = ""):
    """Log API request."""
    logger.debug(f"API {method} {endpoint} {details}")


def log_move_evaluation(fen_short: str, num_moves: int, best_move: str, q_value: float, time_ms: float):
    """Log move evaluation."""
    logger.debug(f"Move Eval: {num_moves} moves analyzed | Best: {best_move} (Q={q_value:.4f}) | Time: {time_ms:.1f}ms")


def log_error(message: str, exception: Optional[Exception] = None):
    """Log error with optional exception details."""
    if exception:
        logger.error(f"{message}: {type(exception).__name__}: {exception}")
    else:
        logger.error(message)


def log_warning(message: str):
    """Log warning."""
    logger.warning(message)


def log_session_summary(total_training_time: float, games_trained: int, final_loss: float):
    """Log complete session summary."""
    logger.info("=" * 60)
    logger.info("SESSION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Training Time: {total_training_time:.2f}s ({total_training_time/60:.1f} min)")
    logger.info(f"Games Trained: {games_trained:,}")
    logger.info(f"Final Loss: {final_loss:.6f}")
    logger.info(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
