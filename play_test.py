# -*- coding: utf-8 -*-
"""
Created on a lonly night before midterm

@author: Allan, Veronica
"""

from game import Game
import numpy as np
import random
import sys
import torch.nn as nn
import torch
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, hidden_dim, drop_out):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_out),
            nn.Linear(hidden_dim, 4)
        )
    def forward(self, x):
        x = x.to(device)
        return self.fc(x)

def play_game(test_game):
    while True:
        print(test_game.available_actions())
        test_game.print_state()
        try:
            x = int(input("action: "))
        except:
            continue
        if x == -1: break
        test_game.do_action(x)
    
def random_play(test_game):
    while not test_game.game_over():
        action = random.choice(test_game.available_actions())
        test_game.do_action(action)
    return test_game.score()
    
def train_game(game, it):
    global losses
    batch_label, batch_output = [], []
    step = 1
    while True:
        Q_values = model(game.vector())
        Q_valid_values = [Q_values[a] if game.is_action_available(a) else float('-inf') for a in range(4)]
        best_action = np.argmax(Q_valid_values)
        reward = game.do_action(best_action)
        Q_star = Q_valid_values[best_action]
        try:
            new_state, vec, reward = game.get_next_state(best_action)
        except:
            print(new_state, vec, reward)
#            sys.exit(0)
        with torch.no_grad():
            Q_next = model(vec)
        batch_output.append(Q_star)
        batch_label.append(reward + gamma * max(Q_next))
        if step % batch_size == 0 or game.game_over():
            if len(batch_label) == 0: return
            optimizer.zero_grad()
            label_tensor = torch.stack(batch_label)
            output_tensor = torch.stack(batch_output)
            batch_label, batch_output = [], []
            loss = criterion(output_tensor, label_tensor)
            loss.backward()
            optimizer.step()
#            print(loss.item())
            losses.append(loss.item())
            if game.game_over():
                print("epoch: {}, game score: {}".format(it, game.score()))
                return
        step += 1
        
def eval_game(game):
    global scores
    model.eval()
    with torch.no_grad():
        for i in range(n_eval):
            game = Game()
            while not game.game_over():
                Q_values = model(game.vector())
                Q_valid_values = [Q_values[a] if game.is_action_available(a) else float('-inf') for a in range(4)]
                best_action = np.argmax(Q_valid_values)
                game.do_action(best_action)
            print("game score: {}".format(game.score()))
            scores.append(game.score())

batch_size = 128
hidden_dim = 128
drop_out = 0.2
n_epoch = 1000
n_eval = 100
gamma = 1
    
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Net(hidden_dim, drop_out)
model = model.to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()
criterion = criterion.to(device)

losses = []
scores = []
randoms = []

if __name__=="__main__":
#    for i in range(100):
#        game = Game()
#        randoms.append(random_play(game))
#    print("the mean of the score is {}".format(np.mean(randoms)))
    model.train()
    for it in range(n_epoch):
        game = Game()
        train_game(game, it)
    eval_game(game)
    print("the mean of the score is {}".format(np.mean(scores)))
        