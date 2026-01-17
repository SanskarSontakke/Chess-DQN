import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import * as ChessJS from 'chess.js';
import { ChessService, StrategicMoveResponse } from './chess.service';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatSelectModule } from '@angular/material/select';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatListModule } from '@angular/material/list';
import { MatSlideToggleModule } from '@angular/material/slide-toggle';
import { MatSliderModule } from '@angular/material/slider';
import { MatTooltipModule } from '@angular/material/tooltip';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatCardModule,
    MatButtonModule,
    MatIconModule,
    MatToolbarModule,
    MatSelectModule,
    MatFormFieldModule,
    MatProgressBarModule,
    MatListModule,
    MatSlideToggleModule,
    MatSliderModule,
    MatTooltipModule
  ],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent implements OnInit {
  game: any;
  board: any[][] = [];
  thoughtProcess: any[] = [];
  evalScore: number = 0;
  isThinking: boolean = false;

  // AI Mode:
  // - 'dqn': Uses trained neural network only (fast, depends on training)
  // - 'hybrid': DQN + Minimax lookahead (best - uses training AND plans ahead)
  // - 'turbo': Minimax with heuristics only (no training needed, always decent)
  aiMode: 'dqn' | 'hybrid' | 'turbo' = 'hybrid';
  searchDepth: number = 3;
  playerColor: 'white' | 'black' = 'white';
  boardFlipped: boolean = false;

  // Strategic thinking display
  strategicData: StrategicMoveResponse | null = null;
  principalVariation: string[] = [];
  gamePhase: string = 'OPENING';

  // Move history
  moveHistory: string[] = [];
  moveHistoryVerbose: any[] = [];

  // Visualization
  highlightedSquares: Map<string, string> = new Map();
  selectedSquare: string | null = null;
  lastMove: { from: string; to: string } | null = null;

  constructor(private chessService: ChessService) {
    try {
      const Chess = (ChessJS as any).Chess || ChessJS;
      this.game = new Chess();
    } catch (e) {
      console.error('Chess init failed', e);
    }
  }

  ngOnInit() {
    if (this.game) {
      this.updateBoard();
    }
  }

  updateBoard() {
    if (!this.game) return;
    this.board = this.game.board();
  }

  get currentTurn(): string {
    return this.game?.turn() === 'w' ? 'White' : 'Black';
  }

  get isPlayerTurn(): boolean {
    const currentTurn = this.game?.turn();
    return (this.playerColor === 'white' && currentTurn === 'w') ||
      (this.playerColor === 'black' && currentTurn === 'b');
  }

  get displayBoard(): any[][] {
    if (this.boardFlipped) {
      return this.board.map(row => [...row].reverse()).reverse();
    }
    return this.board;
  }

  getRowIndex(displayIndex: number): number {
    return this.boardFlipped ? 7 - displayIndex : displayIndex;
  }

  getColIndex(displayIndex: number): number {
    return this.boardFlipped ? 7 - displayIndex : displayIndex;
  }

  onSquareClick(displayRow: number, displayCol: number) {
    if (!this.game || this.isThinking || this.game.isGameOver()) return;
    if (!this.isPlayerTurn) return;

    const row = this.getRowIndex(displayRow);
    const col = this.getColIndex(displayCol);
    const square = String.fromCharCode(97 + col) + (8 - row);
    const piece = this.game.get(square);

    // If a square is already selected, try to move
    if (this.selectedSquare) {
      const moveResult = this.tryMove(this.selectedSquare, square);
      this.clearSelection();

      // If move was successful, let AI respond
      if (moveResult && !this.game.isGameOver()) {
        setTimeout(() => this.makeAIMove(), 300);
      }
      return;
    }

    // Select a piece if it belongs to current player
    const currentTurn = this.game.turn();
    if (piece && piece.color === currentTurn) {
      this.selectedSquare = square;
      this.showLegalMoves(square);
    }
  }

  clearSelection() {
    this.selectedSquare = null;
    this.highlightedSquares.clear();
    if (this.lastMove) {
      this.highlightedSquares.set(this.lastMove.from, 'rgba(255, 255, 100, 0.4)');
      this.highlightedSquares.set(this.lastMove.to, 'rgba(255, 255, 100, 0.4)');
    }
    this.updateBoard();
  }

  showLegalMoves(square: string) {
    this.highlightedSquares.clear();

    // Keep last move highlight
    if (this.lastMove) {
      this.highlightedSquares.set(this.lastMove.from, 'rgba(255, 255, 100, 0.3)');
      this.highlightedSquares.set(this.lastMove.to, 'rgba(255, 255, 100, 0.3)');
    }

    const moves = this.game.moves({ square, verbose: true });
    moves.forEach((move: any) => {
      const isCapture = move.captured;
      this.highlightedSquares.set(move.to, isCapture ? 'rgba(239, 68, 68, 0.5)' : 'rgba(100, 200, 255, 0.4)');
    });

    // Highlight selected square
    this.highlightedSquares.set(square, 'rgba(59, 130, 246, 0.6)');
  }

  tryMove(from: string, to: string): boolean {
    try {
      const move = this.game.move({ from, to, promotion: 'q' });
      if (move) {
        this.moveHistory.push(move.san);
        this.moveHistoryVerbose.push(move);
        this.lastMove = { from, to };
        this.updateBoard();
        this.checkGameOver();
        return true;
      }
      return false;
    } catch (e) {
      return false;
    }
  }

  makeAIMove() {
    if (!this.game || this.game.isGameOver() || this.isThinking) return;

    this.isThinking = true;
    const fen = this.game.fen();

    if (this.aiMode === 'hybrid') {
      // BEST: DQN + Minimax lookahead (trained model evaluates while planning ahead)
      this.chessService.getHybridMove(fen, this.searchDepth).subscribe({
        next: (res) => this.handleAIResponse(res, true),
        error: (err) => {
          console.error('Hybrid move error:', err);
          this.isThinking = false;
        }
      });
    } else if (this.aiMode === 'turbo') {
      // Minimax with handcrafted heuristics (no training needed)
      this.chessService.getStrategicMove(fen, this.searchDepth).subscribe({
        next: (res) => this.handleAIResponse(res, true),
        error: (err) => {
          console.error('Strategic move error:', err);
          this.isThinking = false;
        }
      });
    } else {
      // DQN only: uses trained model for single-move evaluation (fast)
      this.chessService.getMove(fen).subscribe({
        next: (res) => this.handleAIResponse(res, false),
        error: (err) => {
          console.error('DQN move error:', err);
          this.isThinking = false;
        }
      });
    }
  }

  handleAIResponse(res: any, isStrategic: boolean) {
    if (res.best_move) {
      const move = this.game.move(res.best_move);
      if (move) {
        this.moveHistory.push(move.san);
        this.moveHistoryVerbose.push(move);
        this.lastMove = { from: move.from, to: move.to };
      }
      this.updateBoard();

      if (isStrategic) {
        this.strategicData = res;
        this.evalScore = res.score;
        this.principalVariation = res.principal_variation || [];
        this.gamePhase = res.game_phase || 'MIDDLEGAME';
        this.thoughtProcess = (res.candidates || []).map((c: any) => ({
          move: c.san,
          q_value: c.score,
          reasoning: c.reasoning
        }));
      } else {
        this.evalScore = res.q_value;
        this.thoughtProcess = res.thought_process || [];
        this.strategicData = null;
      }

      this.clearSelection();
      this.checkGameOver();
    }
    this.isThinking = false;
  }

  checkGameOver() {
    if (this.game.isGameOver()) {
      setTimeout(() => this.announceGameOver(), 100);
    }
  }

  announceGameOver() {
    let message = 'Game Over! ';
    if (this.game.isCheckmate()) {
      const winner = this.game.turn() === 'w' ? 'Black' : 'White';
      message += `${winner} wins by checkmate!`;
    } else if (this.game.isDraw()) {
      message += 'Draw!';
    } else if (this.game.isStalemate()) {
      message += 'Stalemate!';
    }
    alert(message);
  }

  // Chess Controls
  resetGame() {
    if (this.game) {
      this.game.reset();
      this.updateBoard();
      this.thoughtProcess = [];
      this.evalScore = 0;
      this.strategicData = null;
      this.principalVariation = [];
      this.moveHistory = [];
      this.moveHistoryVerbose = [];
      this.lastMove = null;
      this.clearSelection();
    }
  }

  undoMove() {
    if (this.game && this.moveHistoryVerbose.length >= 2) {
      // Undo two moves (AI + player)
      this.game.undo();
      this.game.undo();
      this.moveHistory.pop();
      this.moveHistory.pop();
      this.moveHistoryVerbose.pop();
      this.moveHistoryVerbose.pop();

      // Update last move
      if (this.moveHistoryVerbose.length > 0) {
        const last = this.moveHistoryVerbose[this.moveHistoryVerbose.length - 1];
        this.lastMove = { from: last.from, to: last.to };
      } else {
        this.lastMove = null;
      }

      this.clearSelection();
      this.updateBoard();
    }
  }

  flipBoard() {
    this.boardFlipped = !this.boardFlipped;
  }

  // Training is now done via the Dashboard (localhost:8000/dashboard)
  // Remove startTraining() - use dashboard instead

  getSquareClass(displayRow: number, displayCol: number): string {
    const row = this.getRowIndex(displayRow);
    const col = this.getColIndex(displayCol);
    const isLight = (row + col) % 2 === 0;
    return isLight ? 'light-square' : 'dark-square';
  }

  getSquareHighlight(displayRow: number, displayCol: number): string | null {
    const row = this.getRowIndex(displayRow);
    const col = this.getColIndex(displayCol);
    const square = String.fromCharCode(97 + col) + (8 - row);
    return this.highlightedSquares.get(square) || null;
  }

  isLegalMoveSquare(displayRow: number, displayCol: number): boolean {
    const row = this.getRowIndex(displayRow);
    const col = this.getColIndex(displayCol);
    const square = String.fromCharCode(97 + col) + (8 - row);
    const highlight = this.highlightedSquares.get(square);
    return highlight !== undefined && square !== this.selectedSquare;
  }

  isCaptureSquare(displayRow: number, displayCol: number): boolean {
    const row = this.getRowIndex(displayRow);
    const col = this.getColIndex(displayCol);
    const square = String.fromCharCode(97 + col) + (8 - row);
    const highlight = this.highlightedSquares.get(square);
    return highlight === 'rgba(239, 68, 68, 0.5)';
  }

  getPieceImage(piece: any): string {
    if (!piece) return '';
    const color = piece.color === 'w' ? 'w' : 'b';
    const type = piece.type.toUpperCase();
    return `https://upload.wikimedia.org/wikipedia/commons/${this.getPieceAsset(color, type)}`;
  }

  private getPieceAsset(color: string, type: string): string {
    const assets: any = {
      'WP': '4/45/Chess_plt45.svg', 'WR': '7/72/Chess_rlt45.svg', 'WN': '7/70/Chess_nlt45.svg',
      'WB': 'b/b1/Chess_blt45.svg', 'WQ': '1/15/Chess_qlt45.svg', 'WK': '4/42/Chess_klt45.svg',
      'BP': 'c/c7/Chess_pdt45.svg', 'BR': 'f/ff/Chess_rdt45.svg', 'BN': 'e/ef/Chess_ndt45.svg',
      'BB': '9/98/Chess_bdt45.svg', 'BQ': '4/47/Chess_qdt45.svg', 'BK': 'f/f0/Chess_kdt45.svg',
    };
    return assets[color.toUpperCase() + type];
  }

  getRankLabel(displayRow: number): string {
    const row = this.getRowIndex(displayRow);
    return String(8 - row);
  }

  getFileLabel(displayCol: number): string {
    const col = this.getColIndex(displayCol);
    return String.fromCharCode(97 + col);
  }

  getMoveNumber(): number {
    return Math.floor(this.moveHistory.length / 2);
  }

  getMoveNumberAt(index: number): number {
    return Math.floor(index / 2) + 1;
  }
}
