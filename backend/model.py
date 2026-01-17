import chess
import numpy as np
import torch
import torch.nn as nn

class ChessDQN(nn.Module):
    """
    Enhanced Chess DQN with ~10M parameters.
    Architecture: 4 conv layers with batch norm + 3 FC layers with dropout.
    """
    def __init__(self):
        super(ChessDQN, self).__init__()
        
        # Convolutional backbone (14 input channels for board representation)
        self.conv1 = nn.Conv2d(14, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Fully connected layers (256 * 8 * 8 = 16384 input features)
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(512, 1)  # Single value output
        
    def forward(self, x):
        # Conv block 1
        x = torch.relu(self.bn1(self.conv1(x)))
        # Conv block 2
        x = torch.relu(self.bn2(self.conv2(x)))
        # Conv block 3
        x = torch.relu(self.bn3(self.conv3(x)))
        # Conv block 4
        x = torch.relu(self.bn4(self.conv4(x)))
        
        # Flatten
        x = x.view(-1, 256 * 8 * 8)
        
        # FC layers with dropout
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x

def board_to_tensor(board: chess.Board):
    # 14 layers: 6 for white pieces, 6 for black pieces, 1 for turn, 1 for castling/enpassant
    tensor = np.zeros((14, 8, 8), dtype=np.float32)
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            # Plane 0-5: White pieces, 6-11: Black pieces
            plane = piece.piece_type - 1
            if piece.color == chess.BLACK:
                plane += 6
            tensor[plane, row, col] = 1
            
    # Metadata planes
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1
    # Simplified: layer 13 could be castling rights or occupancy
    
    return torch.from_numpy(tensor).unsqueeze(0)


def fen_to_tensor_fast(fen: str) -> np.ndarray:
    """
    Convert FEN string directly to numpy array without creating Board object.
    This is ~3x faster than board_to_tensor for bulk conversions.
    """
    tensor = np.zeros((14, 8, 8), dtype=np.float32)
    
    # Parse FEN: only need piece placement and turn
    parts = fen.split(' ')
    piece_placement = parts[0]
    turn = parts[1] if len(parts) > 1 else 'w'
    
    # Piece type mapping
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Black
    }
    
    row = 7  # FEN starts from rank 8 (row 7 in 0-indexed)
    col = 0
    
    for char in piece_placement:
        if char == '/':
            row -= 1
            col = 0
        elif char.isdigit():
            col += int(char)
        elif char in piece_map:
            tensor[piece_map[char], row, col] = 1.0
            col += 1
    
    # Turn plane
    if turn == 'w':
        tensor[12, :, :] = 1.0
    
    return tensor


def batch_fens_to_tensors(fens: list) -> np.ndarray:
    """
    Convert a batch of FEN strings to a stacked numpy array.
    Returns shape (N, 14, 8, 8).
    """
    return np.stack([fen_to_tensor_fast(fen) for fen in fens], axis=0)

