import chess
import random
from model import board_to_tensor

# Standard chess piece values (in centipawns / 100)
PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.25,  # Slightly better than knight in open positions
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0  # King is invaluable, but not counted in material
}


def calculate_material(board: chess.Board, color: chess.Color) -> float:
    """Calculate total material value for a given color."""
    material = 0.0
    for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        material += len(board.pieces(piece_type, color)) * PIECE_VALUES[piece_type]
    return material


def calculate_material_balance(board: chess.Board) -> float:
    """
    Calculate material balance from White's perspective.
    Positive = White is ahead, Negative = Black is ahead.
    """
    white_material = calculate_material(board, chess.WHITE)
    black_material = calculate_material(board, chess.BLACK)
    return white_material - black_material


class ChessEnv:
    def __init__(self):
        self.board = chess.Board()
        self.prev_material_balance = 0.0

    def reset(self):
        self.board.reset()
        self.prev_material_balance = calculate_material_balance(self.board)
        return self.board

    def step(self, move):
        """
        Execute a move and return (board, reward, done).
        
        IMPORTANT: Only legal moves should be passed here.
        python-chess's board.legal_moves already filters out illegal moves,
        including moves that try to use captured pieces.
        
        Reward system:
        - Win: +10
        - Loss: -10
        - Draw: +0.5 (slight positive to avoid bad repetition)
        - Material gain: +value of captured piece (scaled)
        - Material loss: -value of lost piece (scaled)
        - Check: +0.1
        - Checkmate threat: +0.5
        """
        # Validate move is legal (should always be true if using board.legal_moves)
        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move attempted: {move}. This should never happen!")
        
        # Check if this move captures a piece
        captured_piece = self.board.piece_at(move.to_square)
        capture_reward = 0.0
        if captured_piece is not None and captured_piece.color != self.board.turn:
            # Reward for capturing opponent's piece
            capture_reward = PIECE_VALUES.get(captured_piece.piece_type, 0) * 0.1
        
        # Execute move
        self.board.push(move)
        
        # Check game over
        done = self.board.is_game_over()
        reward = 0.0
        
        if done:
            result = self.board.result()
            if result == "1-0":
                reward = 10.0  # White wins
            elif result == "0-1":
                reward = -10.0  # Black wins
            else:
                reward = 0.5  # Draw (stalemate, repetition, etc.)
        else:
            # Intermediate rewards
            
            # Material balance change
            current_balance = calculate_material_balance(self.board)
            material_delta = current_balance - self.prev_material_balance
            
            # Reward/penalize based on whose turn it was
            # If it was White's turn (now Black's), positive delta is good for White
            if not self.board.turn:  # It's now Black's turn, so White just moved
                reward = material_delta * 0.1 + capture_reward
            else:
                reward = -material_delta * 0.1 + capture_reward
            
            # Small bonuses
            if self.board.is_check():
                reward += 0.1  # Bonus for giving check
            
            self.prev_material_balance = current_balance
        
        return self.board, reward, done

    def get_legal_moves(self):
        """
        Returns all legal moves for the current position.
        This automatically excludes:
        - Moving pieces that have been captured
        - Moving pieces that don't exist
        - Moves that would leave the king in check
        """
        return list(self.board.legal_moves)

    def get_random_move(self):
        legal_moves = self.get_legal_moves()
        return random.choice(legal_moves) if legal_moves else None
    
    def is_valid_move(self, move) -> bool:
        """Check if a move is legal in the current position."""
        return move in self.board.legal_moves
