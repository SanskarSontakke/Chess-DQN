"""
Smart Simulation Module for Enhanced Training

Provides three training modes:
1. Random - Fast random move selection (current default)
2. Model-Guided - Uses neural network with epsilon-greedy exploration
3. Hybrid - Minimax lookahead with DQN evaluation at leaf nodes
"""
import chess
import random
import torch
import numpy as np
from typing import List, Tuple
from model import fen_to_tensor_fast


def simulate_game_random() -> List[Tuple[str, float, str, bool]]:
    """
    Fast random simulation. No neural network involved.
    Returns list of (state_fen, reward, next_fen, done) tuples.
    """
    board = chess.Board()
    experiences = []
    done = False
    
    while not done:
        state_fen = board.fen()
        legal_moves = list(board.legal_moves)
        
        # Random policy
        move = random.choice(legal_moves)
        board.push(move)
        
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


def simulate_game_model_guided(
    model: torch.nn.Module,
    device: torch.device,
    epsilon: float = 0.1
) -> List[Tuple[str, float, str, bool]]:
    """
    Model-guided simulation with epsilon-greedy exploration.
    Uses neural network to evaluate positions and select moves.
    
    Args:
        model: The trained DQN model
        device: GPU/CPU device
        epsilon: Exploration rate (0.1 = 10% random moves)
    
    Returns:
        List of (state_fen, reward, next_fen, done) experiences
    """
    board = chess.Board()
    experiences = []
    done = False
    
    model.eval()
    
    with torch.no_grad():
        while not done:
            state_fen = board.fen()
            legal_moves = list(board.legal_moves)
            
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                # Exploration: random move
                move = random.choice(legal_moves)
            else:
                # Exploitation: use model to select best move
                best_move = None
                best_score = float('-inf') if board.turn == chess.WHITE else float('inf')
                
                for candidate_move in legal_moves:
                    # Make the move temporarily
                    board.push(candidate_move)
                    
                    # Evaluate resulting position
                    tensor = torch.from_numpy(fen_to_tensor_fast(board.fen())).unsqueeze(0).to(device)
                    score = model(tensor).item()
                    
                    # Undo move
                    board.pop()
                    
                    # Track best move (maximize for White, minimize for Black)
                    if board.turn == chess.WHITE:
                        if score > best_score:
                            best_score = score
                            best_move = candidate_move
                    else:
                        if score < best_score:
                            best_score = score
                            best_move = candidate_move
                
                move = best_move if best_move else random.choice(legal_moves)
            
            # Execute selected move
            board.push(move)
            
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


def simulate_game_hybrid(
    model: torch.nn.Module,
    device: torch.device,
    depth: int = 2,
    epsilon: float = 0.05
) -> List[Tuple[str, float, str, bool]]:
    """
    Hybrid simulation using Minimax with DQN evaluation at leaf nodes.
    This provides the highest quality training data but is slower.
    
    Args:
        model: The trained DQN model
        device: GPU/CPU device  
        depth: Minimax search depth (2-3 recommended for speed)
        epsilon: Exploration rate (lower for hybrid since moves are already good)
    
    Returns:
        List of (state_fen, reward, next_fen, done) experiences
    """
    board = chess.Board()
    experiences = []
    done = False
    
    model.eval()
    
    def minimax_evaluate(b: chess.Board, d: int, alpha: float, beta: float, maximizing: bool) -> float:
        """Minimax with alpha-beta pruning using DQN for leaf evaluation."""
        if d == 0 or b.is_game_over():
            # Leaf node: use neural network for evaluation
            tensor = torch.from_numpy(fen_to_tensor_fast(b.fen())).unsqueeze(0).to(device)
            with torch.no_grad():
                return model(tensor).item()
        
        legal_moves = list(b.legal_moves)
        
        if maximizing:
            max_eval = float('-inf')
            for m in legal_moves:
                b.push(m)
                eval_score = minimax_evaluate(b, d - 1, alpha, beta, False)
                b.pop()
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for m in legal_moves:
                b.push(m)
                eval_score = minimax_evaluate(b, d - 1, alpha, beta, True)
                b.pop()
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval
    
    with torch.no_grad():
        while not done:
            state_fen = board.fen()
            legal_moves = list(board.legal_moves)
            
            # Epsilon-greedy with lower epsilon (since minimax already provides good moves)
            if random.random() < epsilon:
                move = random.choice(legal_moves)
            else:
                # Find best move using minimax with DQN evaluation
                best_move = None
                best_score = float('-inf') if board.turn == chess.WHITE else float('inf')
                
                for candidate_move in legal_moves:
                    board.push(candidate_move)
                    
                    # Minimax search from this position
                    score = minimax_evaluate(
                        board, 
                        depth - 1, 
                        float('-inf'), 
                        float('inf'), 
                        board.turn == chess.WHITE
                    )
                    
                    board.pop()
                    
                    if board.turn == chess.WHITE:
                        if score > best_score:
                            best_score = score
                            best_move = candidate_move
                    else:
                        if score < best_score:
                            best_score = score
                            best_move = candidate_move
                
                move = best_move if best_move else random.choice(legal_moves)
            
            # Execute selected move
            board.push(move)
            
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
