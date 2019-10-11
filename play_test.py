# -*- coding: utf-8 -*-
"""
Created on a lonly night before midterm

@author: Allan, Veronica
"""

from game import Game
import numpy as np
import random



def play_game():
    while True:
        print(test_game.available_actions())
        test_game.print_state()
        try:
            x = int(input("action: "))
        except:
            continue
        if x == -1: break
        test_game.do_action(x)
    
def random_play():
    while not test_game.game_over():
        action = random.choice(test_game.available_actions())
        test_game.do_action(action)
    test_game.print_state()
    print("final score is: {}".format(test_game.score()))
    
test_game = Game()
random_play()