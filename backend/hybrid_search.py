"""
Hybrid DQN-Minimax Search Engine
Combines neural network evaluation with Minimax lookahead.
The DQN model evaluates leaf positions instead of handcrafted heuristics.
"""
import chess
import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from model import ChessDQN, board_to_tensor
from strategic_eval import get_game_phase, get_full_strategic_evaluation

# Piece values for move ordering only (not evaluation)
PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.25,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0
}


@dataclass
class HybridMoveEvaluation:
    """Stores evaluation details for a single move."""
    move: chess.Move
    score: float
    depth_searched: int
    principal_variation: List[chess.Move]
    reasoning: str


@dataclass
class HybridThinkingLog:
    """Complete log of hybrid search thinking process."""
    position_fen: str
    game_phase: str
    depth: int
    time_spent: float
    nodes_evaluated: int
    model_evaluations: int
    candidate_moves: List[HybridMoveEvaluation]
    best_move: chess.Move
    best_score: float
    principal_variation: List[chess.Move]
    strategic_assessment: Dict[str, float]


class HybridSearchEngine:
    """
    Hybrid DQN-Minimax Search Engine.
    
    Uses Minimax with Alpha-Beta pruning for lookahead,
    but evaluates leaf positions using the trained DQN neural network.
    
    This combines:
    - Strategic planning (looking ahead N moves)
    - Learned evaluation (neural network assesses positions based on training)
    """
    
    def __init__(self, model: ChessDQN, device: torch.device):
        self.model = model
        self.device = device
        self.nodes_evaluated = 0
        self.model_evaluations = 0
        self.thinking_logs: List[HybridThinkingLog] = []
    
    def evaluate_with_nn(self, board: chess.Board) -> float:
        """
        Evaluate a position using the trained neural network.
        Returns score from White's perspective.
        """
        self.model_evaluations += 1
        
        # Handle terminal positions
        if board.is_checkmate():
            return -100.0 if board.turn == chess.WHITE else 100.0
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0
        if board.can_claim_draw():
            return 0.0
        
        # Use neural network for evaluation
        self.model.eval()
        with torch.no_grad():
            state_tensor = board_to_tensor(board).to(self.device)
            q_value = self.model(state_tensor).item()
        
        # Scale Q-value to reasonable chess score range (-10 to +10)
        # Neural network output is typically -1 to 1, scale appropriately
        return q_value * 5.0
    
    def order_moves(self, board: chess.Board, moves: List[chess.Move]) -> List[chess.Move]:
        """Order moves for better alpha-beta pruning."""
        def move_priority(move: chess.Move) -> int:
            score = 0
            
            # Captures (MVV-LVA)
            captured = board.piece_at(move.to_square)
            if captured:
                victim_value = PIECE_VALUES.get(captured.piece_type, 0) * 10
                attacker = board.piece_at(move.from_square)
                attacker_value = PIECE_VALUES.get(attacker.piece_type, 0) if attacker else 0
                score += victim_value - attacker_value + 100
            
            # Checks
            board.push(move)
            if board.is_check():
                score += 50
            board.pop()
            
            # Castling
            if board.is_castling(move):
                score += 40
            
            # Central squares
            to_file = chess.square_file(move.to_square)
            to_rank = chess.square_rank(move.to_square)
            if 2 <= to_file <= 5 and 2 <= to_rank <= 5:
                score += 10
            
            return score
        
        return sorted(moves, key=move_priority, reverse=True)
    
    def get_move_reasoning(self, board: chess.Board, move: chess.Move) -> str:
        """Generate reasoning for a move."""
        reasons = []
        
        captured = board.piece_at(move.to_square)
        if captured:
            reasons.append(f"captures {chess.piece_name(captured.piece_type)}")
        
        board.push(move)
        if board.is_checkmate():
            reasons.append("CHECKMATE!")
        elif board.is_check():
            reasons.append("gives check")
        board.pop()
        
        if board.is_castling(move):
            reasons.append("castles")
        
        moving_piece = board.piece_at(move.from_square)
        if moving_piece:
            if moving_piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                reasons.append("develops piece")
            elif moving_piece.piece_type == chess.PAWN:
                to_sq = move.to_square
                if to_sq in [chess.D4, chess.E4, chess.D5, chess.E5]:
                    reasons.append("central pawn")
        
        return ", ".join(reasons) if reasons else "positional (NN evaluated)"
    
    def minimax_dqn(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        maximizing_white: bool,
        pv: List[chess.Move]
    ) -> Tuple[float, List[chess.Move]]:
        """
        Minimax with Alpha-Beta pruning using DQN for leaf evaluation.
        """
        self.nodes_evaluated += 1
        
        # Leaf node: evaluate with neural network
        if depth == 0 or board.is_game_over():
            return self.evaluate_with_nn(board), pv.copy()
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return self.evaluate_with_nn(board), pv.copy()
        
        ordered_moves = self.order_moves(board, legal_moves)
        best_pv = pv.copy()
        
        if maximizing_white:
            max_eval = float('-inf')
            for move in ordered_moves:
                board.push(move)
                current_pv = pv + [move]
                eval_score, child_pv = self.minimax_dqn(
                    board, depth - 1, alpha, beta, False, current_pv
                )
                board.pop()
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_pv = child_pv
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            
            return max_eval, best_pv
        else:
            min_eval = float('inf')
            for move in ordered_moves:
                board.push(move)
                current_pv = pv + [move]
                eval_score, child_pv = self.minimax_dqn(
                    board, depth - 1, alpha, beta, True, current_pv
                )
                board.pop()
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_pv = child_pv
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            
            return min_eval, best_pv
    
    def search(
        self,
        board: chess.Board,
        depth: int = 3
    ) -> Tuple[chess.Move, HybridThinkingLog]:
        """
        Search for the best move using hybrid DQN-Minimax.
        
        Args:
            board: Current chess position
            depth: Search depth (how many moves to look ahead)
        
        Returns:
            (best_move, thinking_log)
        """
        start_time = time.time()
        self.nodes_evaluated = 0
        self.model_evaluations = 0
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        # Single move: return immediately
        if len(legal_moves) == 1:
            only_move = legal_moves[0]
            log = HybridThinkingLog(
                position_fen=board.fen(),
                game_phase=get_game_phase(board),
                depth=0,
                time_spent=time.time() - start_time,
                nodes_evaluated=1,
                model_evaluations=1,
                candidate_moves=[],
                best_move=only_move,
                best_score=self.evaluate_with_nn(board),
                principal_variation=[only_move],
                strategic_assessment=get_full_strategic_evaluation(board)
            )
            return only_move, log
        
        is_white = board.turn == chess.WHITE
        candidate_evals: List[HybridMoveEvaluation] = []
        
        best_move = legal_moves[0]
        best_score = float('-inf') if is_white else float('inf')
        best_pv: List[chess.Move] = []
        
        ordered_moves = self.order_moves(board, legal_moves)
        
        for move in ordered_moves:
            board.push(move)
            
            score, pv = self.minimax_dqn(
                board,
                depth=depth - 1,
                alpha=float('-inf'),
                beta=float('inf'),
                maximizing_white=not is_white,
                pv=[move]
            )
            
            board.pop()
            
            reasoning = self.get_move_reasoning(board, move)
            
            move_eval = HybridMoveEvaluation(
                move=move,
                score=score,
                depth_searched=depth,
                principal_variation=pv,
                reasoning=reasoning
            )
            candidate_evals.append(move_eval)
            
            if is_white:
                if score > best_score:
                    best_score = score
                    best_move = move
                    best_pv = pv
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
                    best_pv = pv
        
        # Sort candidates
        candidate_evals.sort(key=lambda x: x.score, reverse=is_white)
        
        log = HybridThinkingLog(
            position_fen=board.fen(),
            game_phase=get_game_phase(board),
            depth=depth,
            time_spent=time.time() - start_time,
            nodes_evaluated=self.nodes_evaluated,
            model_evaluations=self.model_evaluations,
            candidate_moves=candidate_evals[:5],
            best_move=best_move,
            best_score=best_score,
            principal_variation=best_pv,
            strategic_assessment=get_full_strategic_evaluation(board)
        )
        
        self.thinking_logs.append(log)
        
        return best_move, log
    
    def format_thinking_log(self, log: HybridThinkingLog) -> str:
        """Format thinking log as human-readable string."""
        lines = [
            f"[HYBRID DQN-MINIMAX] {log.game_phase} phase",
            f"├─ Depth: {log.depth} ply | Time: {log.time_spent:.3f}s",
            f"├─ Nodes: {log.nodes_evaluated} | NN Evals: {log.model_evaluations}",
            f"│"
        ]
        
        if log.candidate_moves:
            lines.append("├─ Top Candidates (DQN evaluated):")
            for i, move_eval in enumerate(log.candidate_moves[:5], 1):
                move_san = chess.Board(log.position_fen).san(move_eval.move)
                sign = "+" if move_eval.score >= 0 else ""
                lines.append(
                    f"│   {i}. {move_san:6s} score: {sign}{move_eval.score:.2f}  ({move_eval.reasoning})"
                )
        
        if log.principal_variation:
            temp_board = chess.Board(log.position_fen)
            pv_san = []
            for move in log.principal_variation[:6]:
                pv_san.append(temp_board.san(move))
                temp_board.push(move)
            lines.append(f"│")
            lines.append(f"├─ Best Line: {' → '.join(pv_san)}")
        
        best_san = chess.Board(log.position_fen).san(log.best_move)
        sign = "+" if log.best_score >= 0 else ""
        lines.append(f"│")
        lines.append(f"└─ Selected: {best_san} ({sign}{log.best_score:.2f})")
        
        return "\n".join(lines)
    
    def get_recent_logs(self, n: int = 5) -> List[HybridThinkingLog]:
        """Get the n most recent thinking logs."""
        return self.thinking_logs[-n:]
    
    def clear_logs(self):
        """Clear all thinking logs."""
        self.thinking_logs.clear()
