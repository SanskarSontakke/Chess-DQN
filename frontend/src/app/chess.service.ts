import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface StrategicMoveResponse {
  best_move: string;
  best_move_san: string;
  score: number;
  depth: number;
  game_phase: string;
  time_ms: number;
  candidates: {
    move: string;
    san: string;
    score: number;
    reasoning: string;
  }[];
  principal_variation: string[];
  strategic_factors: {
    center_control: number;
    piece_development: number;
    king_safety: number;
    pawn_structure: number;
    piece_mobility: number;
  };
  thinking_log: string;
}

export interface BasicMoveResponse {
  best_move: string;
  q_value: number;
  thought_process: {
    move: string;
    q_value: number;
  }[];
}

@Injectable({
  providedIn: 'root'
})
export class ChessService {
  private apiUrl = 'http://localhost:8000';

  constructor(private http: HttpClient) { }

  // Basic DQN move (fast, no lookahead)
  getMove(fen: string): Observable<BasicMoveResponse> {
    return this.http.post<BasicMoveResponse>(`${this.apiUrl}/move`, { fen });
  }

  // Strategic move with Minimax lookahead (slower, smarter)
  getStrategicMove(fen: string, depth: number = 3): Observable<StrategicMoveResponse> {
    return this.http.post<StrategicMoveResponse>(`${this.apiUrl}/strategic-move`, { fen, depth });
  }

  // Hybrid move: DQN neural network + Minimax lookahead (best of both worlds!)
  // Uses the trained model to evaluate positions while looking ahead N moves
  getHybridMove(fen: string, depth: number = 3): Observable<StrategicMoveResponse> {
    return this.http.post<StrategicMoveResponse>(`${this.apiUrl}/hybrid-move`, { fen, depth });
  }

  train(games: number = 100): Observable<any> {
    return this.http.post(`${this.apiUrl}/train`, { games });
  }

  getStatus(): Observable<any> {
    return this.http.get(`${this.apiUrl}/`);
  }

  getTrainingStatus(): Observable<any> {
    return this.http.get(`${this.apiUrl}/training-status`);
  }

  getAIThinking(): Observable<any> {
    return this.http.get(`${this.apiUrl}/ai-thinking`);
  }
}
