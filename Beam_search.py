#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 14:43:29 2019

@author: allan
"""

from game import Game
import numpy as np
import random
import sys

def get_neighbors(game):
    neighbors = [game.copy() for a in game.available_actions()]
    for i, a in enumerate(game.available_actions()): neighbors[i].do_action(a)
    return neighbors

def beam_search(g, width = 16, depth = 5):
    beam = [(a, game) for a, game in zip(g.available_actions(), get_neighbors(g))]
    for i in range(depth):
        neighbors = [(b[0], n) for b in beam for n in get_neighbors(b[1])]
        scores = np.array(list(map(lambda x: x[1].eval(), neighbors)))
        indexes = np.argsort(scores)[-width:]
        beam = np.array(neighbors)[indexes]
    return beam[-1][0] if len(beam) != 0 else g.available_actions()[0]

def play_game(game):
    while not game.game_over():
        game.do_action(beam_search(game, width = search_width, depth = search_depth))
    print("score is: {} max tile is: {}".format(game.score(), game.max_tile()))
    return game.score()


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
search_width = 16
search_depth = 5
scores = []

if __name__=="__main__":
    for i in range(10):
        game = Game()
        scores.append(play_game(game))
    print("the mean of the score is {}".format(np.mean(scores)))