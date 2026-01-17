import torch
import torch.optim as optim
import torch.nn as nn
import random
from collections import deque
from model import ChessDQN, board_to_tensor
import chess
import threading
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "chess_model.pt")

class ReplayBuffer:
    def __init__(self, capacity=100000):  # Increased capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, reward, next_state, done):
        self.buffer.append((state, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    
    def extend(self, items):
        """Add multiple items at once."""
        self.buffer.extend(items)


class Trainer:
    def __init__(self, model, target_model, device, lr=0.001, gamma=0.99, epsilon=0.1):
        self.model = model
        self.target_model = target_model
        self.device = device
        self.lr = lr  # Store for access
        self.epsilon = epsilon  # Store for access
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.memory = ReplayBuffer(capacity=100000)
        self.batch_size = 512  # MUCH larger batch for GPU saturation
        self.lock = threading.Lock()

    def train_step(self, num_updates=1):
        """Run multiple gradient updates per call."""
        if len(self.memory) < self.batch_size:
            return None
        
        total_loss = 0.0
        with self.lock:
            for _ in range(num_updates):
                batch = self.memory.sample(self.batch_size)
                states, rewards, next_states, dones = zip(*batch)

                states = torch.cat(states).to(self.device)
                rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
                next_states = torch.cat(next_states).to(self.device)
                dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

                # Current Q values
                current_q = self.model(states).squeeze()
                
                # Next Q values from target model
                with torch.no_grad():
                    next_q = self.target_model(next_states).squeeze()
                
                # Target Q value
                target_q = rewards + (1 - dones) * self.gamma * next_q

                loss = nn.MSELoss()(current_q, target_q)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
        return total_loss / num_updates if num_updates > 0 else 0.0

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, path=MODEL_PATH):
        """Save model weights to disk."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path=MODEL_PATH):
        """Load model weights from disk. Handles architecture changes gracefully."""
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"Model loaded from {path}")
                return True
            except RuntimeError as e:
                # Architecture mismatch - delete old checkpoint and start fresh
                print(f"⚠️ Model architecture changed! Deleting old checkpoint: {path}")
                print(f"   Reason: {str(e)[:100]}...")
                os.remove(path)
                print("   Starting with fresh random weights.")
                return False
        return False
