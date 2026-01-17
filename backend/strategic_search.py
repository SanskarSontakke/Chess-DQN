"""
Strategic Search Engine for Chess AI
Implements Minimax with Alpha-Beta pruning and strategy logging.
"""
import chess
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from strategic_eval import (
    get_strategic_score, 
    get_full_strategic_evaluation,
    get_game_phase
)

# Piece values for material count
PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.25,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0
}


@dataclass
class MoveEvaluation:
    """Stores evaluation details for a single move."""
    move: chess.Move
    score: float
    depth_searched: int
    principal_variation: List[chess.Move]
    strategic_reasoning: str


@dataclass
class ThinkingLog:
    """Complete log of AI's thinking process."""
    position_fen: str
    game_phase: str
    depth: int
    time_spent: float
    candidate_moves: List[MoveEvaluation]
    best_move: chess.Move
    best_score: float
    principal_variation: List[chess.Move]
    strategic_assessment: Dict[str, float]


def get_material_score(board: chess.Board) -> float:
    """Calculate material balance from White's perspective."""
    score = 0.0
    for piece_type in PIECE_VALUES:
        white_count = len(board.pieces(piece_type, chess.WHITE))
        black_count = len(board.pieces(piece_type, chess.BLACK))
        score += (white_count - black_count) * PIECE_VALUES[piece_type]
    return score


def evaluate_position(board: chess.Board, use_nn: bool = False, model=None) -> float:
    """
    Evaluate a chess position.
    Combines material count with strategic evaluation.
    Can optionally use neural network for evaluation.
    """
    if board.is_checkmate():
        # Checkmate is decisive
        if board.turn == chess.WHITE:
            return -100.0  # Black wins
        return 100.0  # White wins
    
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    
    if board.can_claim_draw():
        return 0.0
    
    # Material score (most important)
    material = get_material_score(board)
    
    # Strategic score (positional factors)
    strategic = get_strategic_score(board)
    
    # Combine: material counts more
    total_score = material + strategic * 0.3
    
    return total_score


def get_move_reasoning(board: chess.Board, move: chess.Move) -> str:
    """Generate strategic reasoning for a move."""
    reasons = []
    
    # Check if it's a capture
    captured = board.piece_at(move.to_square)
    if captured:
        piece_name = chess.piece_name(captured.piece_type)
        reasons.append(f"captures {piece_name}")
    
    # Check if it gives check
    board.push(move)
    if board.is_check():
        reasons.append("gives check")
    if board.is_checkmate():
        reasons.append("CHECKMATE!")
    board.pop()
    
    # Check for development moves
    moving_piece = board.piece_at(move.from_square)
    if moving_piece:
        piece_type = moving_piece.piece_type
        
        if piece_type == chess.KNIGHT:
            # Knight to center is good
            to_file = chess.square_file(move.to_square)
            to_rank = chess.square_rank(move.to_square)
            if 2 <= to_file <= 5 and 2 <= to_rank <= 5:
                reasons.append("knight to center")
        
        elif piece_type == chess.BISHOP:
            # Long diagonal is good
            reasons.append("develops bishop")
        
        elif piece_type == chess.PAWN:
            # Central pawn moves
            to_square = move.to_square
            if to_square in [chess.D4, chess.E4, chess.D5, chess.E5]:
                reasons.append("central pawn")
    
    # Castling
    if board.is_castling(move):
        reasons.append("castles for safety")
    
    return ", ".join(reasons) if reasons else "positional"


def order_moves(board: chess.Board, moves: List[chess.Move]) -> List[chess.Move]:
    """
    Order moves for better alpha-beta pruning.
    Better moves first = more pruning.
    """
    def move_priority(move: chess.Move) -> int:
        score = 0
        
        # Captures are good to try first
        captured = board.piece_at(move.to_square)
        if captured:
            # MVV-LVA: Most Valuable Victim - Least Valuable Attacker
            victim_value = PIECE_VALUES.get(captured.piece_type, 0) * 10
            attacker = board.piece_at(move.from_square)
            attacker_value = PIECE_VALUES.get(attacker.piece_type, 0) if attacker else 0
            score += victim_value - attacker_value + 100
        
        # Checks are important
        board.push(move)
        if board.is_check():
            score += 50
        board.pop()
        
        # Castling is usually good
        if board.is_castling(move):
            score += 40
        
        # Central squares
        to_file = chess.square_file(move.to_square)
        to_rank = chess.square_rank(move.to_square)
        if 2 <= to_file <= 5 and 2 <= to_rank <= 5:
            score += 10
        
        return score
    
    return sorted(moves, key=move_priority, reverse=True)


def minimax_alpha_beta(
    board: chess.Board,
    depth: int,
    alpha: float,
    beta: float,
    maximizing_white: bool,
    pv: List[chess.Move]  # Principal variation
) -> Tuple[float, List[chess.Move]]:
    """
    Minimax search with alpha-beta pruning.
    Returns (score, principal_variation).
    """
    if depth == 0 or board.is_game_over():
        return evaluate_position(board), pv.copy()
    
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return evaluate_position(board), pv.copy()
    
    # Order moves for better pruning
    ordered_moves = order_moves(board, legal_moves)
    best_pv = pv.copy()
    
    if maximizing_white:
        max_eval = float('-inf')
        for move in ordered_moves:
            board.push(move)
            current_pv = pv + [move]
            eval_score, child_pv = minimax_alpha_beta(
                board, depth - 1, alpha, beta, False, current_pv
            )
            board.pop()
            
            if eval_score > max_eval:
                max_eval = eval_score
                best_pv = child_pv
            
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # Beta cutoff
        
        return max_eval, best_pv
    else:
        min_eval = float('inf')
        for move in ordered_moves:
            board.push(move)
            current_pv = pv + [move]
            eval_score, child_pv = minimax_alpha_beta(
                board, depth - 1, alpha, beta, True, current_pv
            )
            board.pop()
            
            if eval_score < min_eval:
                min_eval = eval_score
                best_pv = child_pv
            
            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # Alpha cutoff
        
        return min_eval, best_pv


class StrategicSearchEngine:
    """
    AI Search Engine with strategic thinking and logging.
    """
    
    def __init__(self, default_depth: int = 3):
        self.default_depth = default_depth
        self.thinking_logs: List[ThinkingLog] = []
    
    def search(
        self, 
        board: chess.Board, 
        depth: Optional[int] = None,
        time_limit: Optional[float] = None
    ) -> Tuple[chess.Move, ThinkingLog]:
        """
        Search for the best move with full thinking log.
        
        Args:
            board: Current chess position
            depth: Search depth (default: 3)
            time_limit: Max time in seconds (overrides depth with iterative deepening)
        
        Returns:
            (best_move, thinking_log)
        """
        start_time = time.time()
        search_depth = depth or self.default_depth
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        if len(legal_moves) == 1:
            # Only one move, return it immediately
            only_move = legal_moves[0]
            log = ThinkingLog(
                position_fen=board.fen(),
                game_phase=get_game_phase(board),
                depth=0,
                time_spent=time.time() - start_time,
                candidate_moves=[],
                best_move=only_move,
                best_score=0.0,
                principal_variation=[only_move],
                strategic_assessment=get_full_strategic_evaluation(board)
            )
            return only_move, log
        
        # Evaluate all candidate moves
        is_white = board.turn == chess.WHITE
        candidate_evals: List[MoveEvaluation] = []
        
        best_move = legal_moves[0]
        best_score = float('-inf') if is_white else float('inf')
        best_pv: List[chess.Move] = []
        
        # Order moves for efficiency
        ordered_moves = order_moves(board, legal_moves)
        
        for move in ordered_moves:
            board.push(move)
            
            score, pv = minimax_alpha_beta(
                board,
                depth=search_depth - 1,
                alpha=float('-inf'),
                beta=float('inf'),
                maximizing_white=not is_white,  # Opponent's turn after our move
                pv=[move]
            )
            
            board.pop()
            
            reasoning = get_move_reasoning(board, move)
            
            move_eval = MoveEvaluation(
                move=move,
                score=score,
                depth_searched=search_depth,
                principal_variation=pv,
                strategic_reasoning=reasoning
            )
            candidate_evals.append(move_eval)
            
            # Track best move
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
        
        # Sort candidates by score
        candidate_evals.sort(
            key=lambda x: x.score, 
            reverse=is_white
        )
        
        # Create thinking log
        log = ThinkingLog(
            position_fen=board.fen(),
            game_phase=get_game_phase(board),
            depth=search_depth,
            time_spent=time.time() - start_time,
            candidate_moves=candidate_evals[:5],  # Top 5 moves
            best_move=best_move,
            best_score=best_score,
            principal_variation=best_pv,
            strategic_assessment=get_full_strategic_evaluation(board)
        )
        
        self.thinking_logs.append(log)
        
        return best_move, log
    
    def format_thinking_log(self, log: ThinkingLog) -> str:
        """Format thinking log as human-readable string."""
        lines = [
            f"[AI THINKING] {log.game_phase} phase",
            f"├─ Depth: {log.depth} ply | Time: {log.time_spent:.3f}s",
            f"│"
        ]
        
        if log.candidate_moves:
            lines.append("├─ Top Candidates:")
            for i, move_eval in enumerate(log.candidate_moves[:5], 1):
                move_san = chess.Board(log.position_fen).san(move_eval.move)
                reasoning = move_eval.strategic_reasoning or "positional"
                sign = "+" if move_eval.score >= 0 else ""
                lines.append(
                    f"│   {i}. {move_san:6s} score: {sign}{move_eval.score:.2f}  ({reasoning})"
                )
        
        if log.principal_variation:
            temp_board = chess.Board(log.position_fen)
            pv_san = []
            for move in log.principal_variation[:6]:
                pv_san.append(temp_board.san(move))
                temp_board.push(move)
            lines.append(f"│")
            lines.append(f"├─ Principal Variation: {' '.join(pv_san)}")
        
        if log.strategic_assessment:
            lines.append(f"│")
            lines.append(f"├─ Strategic Factors:")
            for factor, value in log.strategic_assessment.items():
                sign = "+" if value >= 0 else ""
                lines.append(f"│   {factor}: {sign}{value:.2f}")
        
        best_san = chess.Board(log.position_fen).san(log.best_move)
        sign = "+" if log.best_score >= 0 else ""
        lines.append(f"│")
        lines.append(f"└─ Selected: {best_san} ({sign}{log.best_score:.2f})")
        
        return "\n".join(lines)
    
    def get_recent_logs(self, n: int = 5) -> List[ThinkingLog]:
        """Get the n most recent thinking logs."""
        return self.thinking_logs[-n:]
    
    def clear_logs(self):
        """Clear all thinking logs."""
        self.thinking_logs.clear()


# Global search engine instance
search_engine = StrategicSearchEngine(default_depth=3)


def get_best_move_with_thinking(board: chess.Board, depth: int = 3) -> Tuple[chess.Move, str]:
    """
    Convenience function to get best move with formatted thinking log.
    Returns (best_move, formatted_log_string).
    """
    move, log = search_engine.search(board, depth=depth)
    formatted = search_engine.format_thinking_log(log)
    return move, formatted
