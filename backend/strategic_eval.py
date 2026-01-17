"""
Strategic Position Evaluation for Chess AI
Provides detailed positional assessment beyond simple material counting.
"""
import chess
from typing import Dict, Tuple, List

# Center squares - most important for control
CENTER_SQUARES = [chess.D4, chess.D5, chess.E4, chess.E5]
EXTENDED_CENTER = [chess.C3, chess.C4, chess.C5, chess.C6, 
                   chess.D3, chess.D6, chess.E3, chess.E6,
                   chess.F3, chess.F4, chess.F5, chess.F6]

# Development squares for minor pieces
WHITE_DEVELOPED = {
    chess.KNIGHT: [chess.C3, chess.F3, chess.D2, chess.E2],
    chess.BISHOP: [chess.C4, chess.B5, chess.G2, chess.B2, chess.E2, chess.D3]
}
BLACK_DEVELOPED = {
    chess.KNIGHT: [chess.C6, chess.F6, chess.D7, chess.E7],
    chess.BISHOP: [chess.C5, chess.B4, chess.G7, chess.B7, chess.E7, chess.D6]
}

# King safety zones
WHITE_KINGSIDE_CASTLE = [chess.F1, chess.G1, chess.H1, chess.F2, chess.G2, chess.H2]
WHITE_QUEENSIDE_CASTLE = [chess.A1, chess.B1, chess.C1, chess.A2, chess.B2, chess.C2]
BLACK_KINGSIDE_CASTLE = [chess.F8, chess.G8, chess.H8, chess.F7, chess.G7, chess.H7]
BLACK_QUEENSIDE_CASTLE = [chess.A8, chess.B8, chess.C8, chess.A7, chess.B7, chess.C7]


def evaluate_center_control(board: chess.Board) -> float:
    """
    Evaluate control of the center squares.
    Returns score from White's perspective.
    """
    score = 0.0
    
    for square in CENTER_SQUARES:
        # Piece occupation (strong)
        piece = board.piece_at(square)
        if piece:
            if piece.color == chess.WHITE:
                score += 0.3
            else:
                score -= 0.3
        
        # Attack control
        white_attackers = len(board.attackers(chess.WHITE, square))
        black_attackers = len(board.attackers(chess.BLACK, square))
        score += (white_attackers - black_attackers) * 0.1
    
    # Extended center (less important)
    for square in EXTENDED_CENTER:
        piece = board.piece_at(square)
        if piece:
            if piece.color == chess.WHITE:
                score += 0.1
            else:
                score -= 0.1
    
    return score


def evaluate_piece_development(board: chess.Board) -> float:
    """
    Evaluate how well pieces are developed.
    Penalizes pieces still on starting squares, rewards active placement.
    """
    score = 0.0
    
    # White knight development
    for sq in [chess.B1, chess.G1]:  # Starting squares
        piece = board.piece_at(sq)
        if piece and piece.piece_type == chess.KNIGHT and piece.color == chess.WHITE:
            score -= 0.2  # Undeveloped penalty
    
    for sq in WHITE_DEVELOPED[chess.KNIGHT]:
        piece = board.piece_at(sq)
        if piece and piece.piece_type == chess.KNIGHT and piece.color == chess.WHITE:
            score += 0.15  # Development bonus
    
    # White bishop development
    for sq in [chess.C1, chess.F1]:
        piece = board.piece_at(sq)
        if piece and piece.piece_type == chess.BISHOP and piece.color == chess.WHITE:
            score -= 0.2
    
    # Black knight development  
    for sq in [chess.B8, chess.G8]:
        piece = board.piece_at(sq)
        if piece and piece.piece_type == chess.KNIGHT and piece.color == chess.BLACK:
            score += 0.2  # Reward for opponent's undeveloped pieces
    
    for sq in BLACK_DEVELOPED[chess.KNIGHT]:
        piece = board.piece_at(sq)
        if piece and piece.piece_type == chess.KNIGHT and piece.color == chess.BLACK:
            score -= 0.15
    
    # Black bishop development
    for sq in [chess.C8, chess.F8]:
        piece = board.piece_at(sq)
        if piece and piece.piece_type == chess.BISHOP and piece.color == chess.BLACK:
            score += 0.2
    
    return score


def evaluate_king_safety(board: chess.Board) -> float:
    """
    Evaluate king safety based on:
    - Castling status
    - Pawn shield in front of king
    - Attacking pieces near king
    """
    score = 0.0
    
    white_king_sq = board.king(chess.WHITE)
    black_king_sq = board.king(chess.BLACK)
    
    if white_king_sq is None or black_king_sq is None:
        return 0.0  # No king found (shouldn't happen)
    
    # Castling bonus
    if board.has_kingside_castling_rights(chess.WHITE) or board.has_queenside_castling_rights(chess.WHITE):
        score += 0.1  # Can still castle
    if white_king_sq in [chess.G1, chess.C1]:  # Has castled
        score += 0.3
    
    if board.has_kingside_castling_rights(chess.BLACK) or board.has_queenside_castling_rights(chess.BLACK):
        score -= 0.1
    if black_king_sq in [chess.G8, chess.C8]:
        score -= 0.3
    
    # Pawn shield for White
    white_king_file = chess.square_file(white_king_sq)
    white_king_rank = chess.square_rank(white_king_sq)
    pawn_shield = 0
    for df in [-1, 0, 1]:
        f = white_king_file + df
        if 0 <= f <= 7:
            for r in [white_king_rank + 1, white_king_rank + 2]:
                if 0 <= r <= 7:
                    sq = chess.square(f, r)
                    piece = board.piece_at(sq)
                    if piece and piece.piece_type == chess.PAWN and piece.color == chess.WHITE:
                        pawn_shield += 1
    score += pawn_shield * 0.1
    
    # Pawn shield for Black
    black_king_file = chess.square_file(black_king_sq)
    black_king_rank = chess.square_rank(black_king_sq)
    pawn_shield = 0
    for df in [-1, 0, 1]:
        f = black_king_file + df
        if 0 <= f <= 7:
            for r in [black_king_rank - 1, black_king_rank - 2]:
                if 0 <= r <= 7:
                    sq = chess.square(f, r)
                    piece = board.piece_at(sq)
                    if piece and piece.piece_type == chess.PAWN and piece.color == chess.BLACK:
                        pawn_shield += 1
    score -= pawn_shield * 0.1
    
    # King exposure (attackers near king)
    white_king_attackers = len(board.attackers(chess.BLACK, white_king_sq))
    black_king_attackers = len(board.attackers(chess.WHITE, black_king_sq))
    score -= white_king_attackers * 0.2
    score += black_king_attackers * 0.2
    
    return score


def evaluate_pawn_structure(board: chess.Board) -> float:
    """
    Evaluate pawn structure quality:
    - Penalize doubled pawns
    - Penalize isolated pawns
    - Bonus for passed pawns
    - Bonus for connected pawns
    """
    score = 0.0
    
    white_pawns = list(board.pieces(chess.PAWN, chess.WHITE))
    black_pawns = list(board.pieces(chess.PAWN, chess.BLACK))
    
    # Analyze White pawns
    white_files = [chess.square_file(sq) for sq in white_pawns]
    for file in range(8):
        count = white_files.count(file)
        if count > 1:
            score -= 0.15 * (count - 1)  # Doubled pawn penalty
    
    for sq in white_pawns:
        file = chess.square_file(sq)
        # Check for isolated pawn
        has_neighbor = any(chess.square_file(p) in [file - 1, file + 1] for p in white_pawns)
        if not has_neighbor:
            score -= 0.1  # Isolated pawn penalty
        
        # Check for passed pawn
        rank = chess.square_rank(sq)
        is_passed = True
        for r in range(rank + 1, 8):
            for df in [-1, 0, 1]:
                f = file + df
                if 0 <= f <= 7:
                    check_sq = chess.square(f, r)
                    piece = board.piece_at(check_sq)
                    if piece and piece.piece_type == chess.PAWN and piece.color == chess.BLACK:
                        is_passed = False
                        break
            if not is_passed:
                break
        if is_passed:
            score += 0.2 + (rank * 0.05)  # Passed pawn bonus, more for advanced
    
    # Analyze Black pawns
    black_files = [chess.square_file(sq) for sq in black_pawns]
    for file in range(8):
        count = black_files.count(file)
        if count > 1:
            score += 0.15 * (count - 1)  # Opponent's doubled pawns is good
    
    for sq in black_pawns:
        file = chess.square_file(sq)
        has_neighbor = any(chess.square_file(p) in [file - 1, file + 1] for p in black_pawns)
        if not has_neighbor:
            score += 0.1
        
        # Check for passed pawn
        rank = chess.square_rank(sq)
        is_passed = True
        for r in range(rank - 1, -1, -1):
            for df in [-1, 0, 1]:
                f = file + df
                if 0 <= f <= 7:
                    check_sq = chess.square(f, r)
                    piece = board.piece_at(check_sq)
                    if piece and piece.piece_type == chess.PAWN and piece.color == chess.WHITE:
                        is_passed = False
                        break
            if not is_passed:
                break
        if is_passed:
            score -= 0.2 + ((7 - rank) * 0.05)
    
    return score


def evaluate_piece_mobility(board: chess.Board) -> float:
    """
    Evaluate piece mobility (number of legal moves for pieces).
    More mobility = better piece activity.
    """
    # Save current turn
    original_turn = board.turn
    
    # Count White's mobility
    board.turn = chess.WHITE
    white_moves = len(list(board.legal_moves))
    
    # Count Black's mobility
    board.turn = chess.BLACK
    black_moves = len(list(board.legal_moves))
    
    # Restore turn
    board.turn = original_turn
    
    # Normalize mobility difference
    return (white_moves - black_moves) * 0.02


def get_full_strategic_evaluation(board: chess.Board) -> Dict[str, float]:
    """
    Get a complete strategic evaluation breakdown.
    Returns dict with individual component scores.
    """
    return {
        "center_control": evaluate_center_control(board),
        "piece_development": evaluate_piece_development(board),
        "king_safety": evaluate_king_safety(board),
        "pawn_structure": evaluate_pawn_structure(board),
        "piece_mobility": evaluate_piece_mobility(board)
    }


def get_strategic_score(board: chess.Board) -> float:
    """
    Get total strategic evaluation score.
    Combines all strategic features into one score.
    """
    components = get_full_strategic_evaluation(board)
    return sum(components.values())


def get_game_phase(board: chess.Board) -> str:
    """
    Determine current game phase: OPENING, MIDDLEGAME, or ENDGAME.
    Based on material and piece development.
    """
    # Count total non-pawn material
    total_pieces = 0
    for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        total_pieces += len(board.pieces(piece_type, chess.WHITE))
        total_pieces += len(board.pieces(piece_type, chess.BLACK))
    
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
    
    if board.fullmove_number <= 10 and total_pieces >= 12:
        return "OPENING"
    elif total_pieces <= 6 or (queens == 0 and total_pieces <= 8):
        return "ENDGAME"
    else:
        return "MIDDLEGAME"
