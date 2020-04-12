# %%
import chess
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
import torch
from torch import autograd
import copy
from collections import deque
import random

# from Qnetwork import QNetwork
start_board = chess.Board()
from boardrep import initial_board_state

TARGET_update = 20

Colour = bool
COLOURS = [WHITE, BLACK] = [True, False]
pieces_dict = {'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6, 'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6}
# chessnet = QNetwork(state_size= 65, action_size= 4032, seed=99)
from Qnetwork2 import ConvDQN
from Qnetwork2 import BasicBuffer
from Qnetwork2 import DQNAgent

target_net = DQNAgent()
alexander = DQNAgent()
#alexander.model.load_state_dict(torch.load('net_after29'))




#%% #movedictionary

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
NUM = [1, 2, 3, 4, 5, 6, 7, 8]
p = ['', 'N', 'B', 'R', 'Q', 'K']
promotions = ['q', 'r', 'b', 'n']

squares = []
for i in range(len(letters)):
    for j in range(len(NUM)):
        s = letters[i] + str(NUM[j])
        squares.append(s)

moves = []

for i in squares:
    for j in squares:
        if i != j:
            n = i + j
            moves.append(n)

promotional_moves = []
for i in range(len(letters)):
    for k in range(len(letters)):
        for j in range(len(promotions)):
            w = letters[i] + str(7) + letters[k] + str(8) + promotions[j]
            b = letters[i] + str(2) + letters[k] + str(1) + promotions[j]

            promotional_moves.append(w)
            promotional_moves.append(b)
#print(promotional_moves)

for item in promotional_moves:
    moves.append(item)
#print(len(moves))
#print('no of p moves^)')

move_no = []
for i in range(len(moves)):
    move_no.append(i)

move_dictionary = {}
for key in moves:
    for value in move_no:
        move_dictionary[key] = value
        move_no.remove(value)
        break




# %%
class ChessAgent1:
    def __init__(self, colour):
        # Global graph variables
        self.board = chess.Board()
        #print(self.board)
        #print('self.board ^')
        self.colour = colour
        self.experience = deque(maxlen=100)
        # agent info

        self.total_reward = 0.
        self.epsilon = None
        # description of agent's state:
        self.current_state = initial_board_state

        # analytics
        self.analytics = {'episode_rewards': [],
                          'epsilon_values': [],
                          'length of game': []}

    def board_to_vec(self):
        board_vec = []
        for square in chess.SQUARES:
            if self.board.piece_at(square) is None:
                board_vec.append(0)
            else:
                board_vec.append(pieces_dict[self.board.piece_at(square).symbol()])
        return board_vec

    def getKeysByValue(self, valueToFind):
        listOfKeys = list()
        listOfItems = move_dictionary.items()
        for item in listOfItems:
            if item[1] == valueToFind:
                listOfKeys.append(item[0])
                break
        return listOfKeys

    def listPossibleActions(self, board):
        actions = self.board.legal_moves
        actions = list(actions)
        action_no = []
        for i in range(len(actions)):
            action_no.append(move_dictionary[str(actions[i])])
        return action_no

    def is_opponent_checkmated(self):
        c = self.whatcolourami()[0]
        k = -6 * c
        if self.board.is_checkmate() is False:
            return False
        else:
            for square in chess.SQUARES:
                if pieces_dict[self.board.piece_at(square).symbol()] == k and len(
                        self.board.attackers(whatcolourami()[1], square)) >= 1:
                    return True

            return False

    def transitionState(self, board, action):
        """
        :param action:
        :param state: current state of agent.
        :return: state after taking action to move agent to node "new_node" when in state "state".
        """
        # print('before move dict')
        # print(action)
        # print('^')
        action = self.getKeysByValue(valueToFind=action)
        # print('after move dict')
        # print(action)
        # print('^')
        action = action[0]
        # print(action)
        self.board.push_uci(action)
        #print(self.board)
        new_state = self.board_to_vec()
        new_state = np.array(new_state)

        return new_state

    def state_of_board(self):
        return self.board_to_vec()

    def whatcolourami(self):
        if self.colour == 1:
            return 1, chess.WHITE
        else:
            return -1, chess.BLACK

    def calculateReward(self):
        """
        if old state
        :param selected_action:
        :param new_state:
        :return:
        """

        if self.board.is_game_over() is False:  ### game continues
            return 0
        else:  #### if i am white and have checkmated black
            if self.board.is_checkmate():
                return 1000

            else:
                return 0

    def isStateTerminal(self):
        if self.board.is_game_over():
            return True
        else:
            return False

    # def updateQnetwork:
    def QLEARNING(self, num_episodes, initial_epsilon):
        print('STARTING q LEARNING')
        self.epsilon = initial_epsilon
        decay_counter = 0
        for episode in range(num_episodes):
            print('episode:', episode)
            self.analytics['epsilon_values'].append(self.epsilon)

            # reset agent/environment state
            self.total_reward = 0.
            self.board.reset()
            boardvec = self.board_to_vec()
            self.current_state = np.array(boardvec)

            if episode == decay_counter:
                decay = True
                # decay epsilon every 5 episodes
                decay_counter += 10
            else:
                decay = False

            for t in itertools.count():
                # select action using policy
                state = self.current_state
                score = sum(state)

                action = self.greedypolicy2(decay=decay, decay_rate=0.01)
                #print('**************')
                new_state = self.transitionState(board=self.board,
                                                 action=action)
                new_score = sum(new_state)
                #print('new score:', new_score)
                #print('old score:', score)
                #print# updates state using old board and new action
                # this function also udates self.board
                #print(self.board)
                #print('action:', action, self.getKeysByValue(action))
                #print('************')

                # Reward function
                reward = self.calculateReward() #+ new_score - score

                # Q update
                # best_next_action = self.bestaction(board = board)[0]
                if self.isStateTerminal():
                    done = 1
                else:
                    done = 0
                #print(state, action, reward, new_state, done)
                batch = state, action, np.array([reward]), new_state, done
                self.experience.append(batch)

                #print('buffer:', alexander.replay_buffer.push(state, action, reward, new_state, done))
                alexander.experience = self.experience
                if len(self.experience) >= 64:
                    alexander.update(64)
                    print('alexanderupating', 't:', t)
                    for name, param in alexander.model.state_dict().items():
                        print(name, param.data)

                # Update state & rewards
                self.total_reward += reward + new_score - score
                # print(self.total_reward)
                #print('t: ', t)
                #print('episode', episode)
                self.current_state = new_state

                if t > 50:
                    self.analytics['length of game'].append(t)
                    self.analytics['episode_rewards'].append(self.total_reward)
                    break


                # Terminate episode if current is terminal
                if self.isStateTerminal():

                    self.analytics['length of game'].append(t)
                    self.analytics['episode_rewards'].append(self.total_reward)
                    print(self.board, t)
                    if episode % TARGET_update == 0:
                        target_net.model.load_state_dict(alexander.model.state_dict())
                    if episode + 1 == num_episodes:
                        #path_name = 'net_after' + str(episode)
                        torch.save(target_net.model.state_dict(), '1000games')
                    break

            print('lenght of games:', self.analytics['length of game'])
            print('episode rewards', self.analytics['episode_rewards'])

    def greedypolicy2(self, decay_rate, decay):  # need to encode 'available'actions
        state = self.board_to_vec()
        state = np.array(state)
        state = autograd.Variable(torch.from_numpy(state).float().unsqueeze(0))
        qval = alexander.model.forward(state)
        print('qval:', qval, qval.size())
        legal_actions = []
        for i in self.listPossibleActions(board=self.board):
            legal_actions.append(i)
        legal_scores = []
        for i in legal_actions:
            legal_scores.append(qval[0][i])
        j = np.argmax(legal_scores)
        best_legal_action = legal_actions[j]

        r = random.uniform(0, 1)
        if r > self.epsilon:
            selected_action = best_legal_action
        else:
            random.shuffle(legal_actions)
            selected_action = legal_actions[0]
        if decay:
            if self.epsilon >= 0.5:
                self.epsilon = self.epsilon * (1. - decay_rate)
            if 0.5 > self.epsilon > 0.15:
                self.epsilon = self.epsilon * (1. - decay_rate / 10.)

            else:
                self.epsilon = self.epsilon

        return selected_action

    def bestaction(self, state):
        state_vector = self.board_to_vec()
        qval = alexander.model(state_vector)
        legal_actions = []
        for i in self.listPossibleActions(board=self.board):
            legal_actions.append(move_dictionary[str(i)])
        legal_scores = []
        for i in legal_actions:
            legal_scores.append(qval[i])
        j = np.argmax(legal_scores)
        best_legal_action = legal_actions[j]
        return best_legal_action

    def plotRewards(self, save_name=None):
        plt.figure()
        plt.xlabel('$Episodes$')
        plt.ylabel('$Total \; Reward$')
        plt.ylim()
        plt.xlim(0, len(self.analytics['episode_rewards']))
        plt.plot(self.analytics['episode_rewards'], 'o-', markersize=0)
        plt.xticks([i for i in
                    np.arange(0, len(self.analytics['episode_rewards']), len(self.analytics['episode_rewards']) / 10)])
        plt.grid(linestyle='--', which='both', axis='both')
        if save_name:
            if not os.path.exists('./plots/'):
                os.mkdir('./plots/')
            plt.savefig(fname='./plots/' + save_name + '.png', dpi=800)
        plt.show()

    def plotEpsilon(self, save_name=None):
        plt.figure()
        plt.xlabel('$Episodes$')
        plt.ylabel('$Epsilon$')
        plt.ylim()
        plt.xlim(0, len(self.analytics['epsilon_values']))
        plt.plot(self.analytics['epsilon_values'], 'o-', markersize=0)
        plt.xticks([i for i in np.arange(0, len(self.analytics['epsilon_values']),
                                         len(self.analytics['epsilon_values']) / 10)])
        plt.grid(linestyle='--', which='both', axis='both')
        if save_name:
            if not os.path.exists('./plots/'):
                os.mkdir('./plots/')
            plt.savefig(fname='./plots/' + save_name + '.png', dpi=800)
        plt.show()


    def plotgamelength(self, save_name=None):
        plt.figure()
        plt.xlabel('$game$')
        plt.ylabel('$gamelength$')
        plt.ylim()
        plt.xlim(0, len(self.analytics['length of game']))
        plt.plot(self.analytics['length of game'], 'o-', markersize=0)
        plt.xticks([i for i in
                    np.arange(0, len(self.analytics['length of game']), len(self.analytics['length of game']) / 10)])
        plt.grid(linestyle='--', which='both', axis='both')
        if save_name:
            if not os.path.exists('./plots/'):
                os.mkdir('./plots/')
            plt.savefig(fname='./plots/' + save_name + '.png', dpi=800)
        plt.show()

# %%
