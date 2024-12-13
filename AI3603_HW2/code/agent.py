"""
This module defines various agent classes for a game, including random agents, greedy agents.
You need to implement your own agent in the YourAgent class using minimax algorithms.

Classes:
    Agent: Base class for all agents.
    RandomAgent: Agent that selects actions randomly.
    SimpleGreedyAgent: Greedy agent that selects actions based on maximum vertical advance.
    YourAgent: Placeholder for user-defined agent.

Class Agent:
    Methods:
        __init__(self, game): Initializes the agent with the game instance.
        getAction(self, state): Abstract method to get the action for the current state.
        oppAction(self, state): Abstract method to get the opponent's action for the current state.

Class RandomAgent(Agent):
    Methods:
        getAction(self, state): Selects a random legal action.
        oppAction(self, state): Selects a random legal action for the opponent.

Class SimpleGreedyAgent(Agent):
    Methods:
        getAction(self, state): Selects an action with the maximum vertical advance.
        oppAction(self, state): Selects an action with the minimum vertical advance for the opponent.

Class YourAgent(Agent):
    Methods:
        getAction(self, state): Placeholder for user-defined action selection.
        oppAction(self, state): Placeholder for user-defined opponent action selection.
"""

import random, re, datetime
from board import Board
import copy


class Agent(object):
    def __init__(self, game):
        self.game = game
        self.action = None
        self.depth = 3
        self.action_list = []
        self.iter = 0
        self.max_iteration = 200

    def getAction(self, state):
        raise Exception("Not implemented yet")

    def oppAction(self, state):
        raise Exception("Not implemented yet")  
    


class RandomAgent(Agent):

    def getAction(self, state):
        legal_actions = self.game.actions(state)
        self.action = random.choice(legal_actions)

    def oppAction(self, state):
        legal_actions = self.game.actions(state)
        self.opp_action = random.choice(legal_actions)


class SimpleGreedyAgent(Agent):
    # a one-step-lookahead greedy agent that returns action with max vertical advance

    def getAction(self, state):

        legal_actions = self.game.actions(state)

        self.action = random.choice(legal_actions)

        player = self.game.player(state)
        if player == 1:
            max_vertical_advance_one_step = max([action[0][0] - action[1][0] for action in legal_actions])
            max_actions = [action for action in legal_actions if action[0][0] - action[1][0] == max_vertical_advance_one_step]
        else:
            max_vertical_advance_one_step = max([action[1][0] - action[0][0] for action in legal_actions])
            max_actions = [action for action in legal_actions if action[1][0] - action[0][0] == max_vertical_advance_one_step]
        self.action = random.choice(max_actions)

    def oppAction(self, state):
        legal_actions = self.game.actions(state)

        self.opp_action = random.choice(legal_actions)

        player = self.game.player(state)
        if player == 1:
            min_vertical_advance_one_step = min([action[0][0] - action[1][0] for action in legal_actions])
            min_actions = [action for action in legal_actions if action[0][0] - action[1][0] == min_vertical_advance_one_step]
        else:
            min_vertical_advance_one_step = min([action[1][0] - action[0][0] for action in legal_actions])
            min_actions = [action for action in legal_actions if action[1][0] - action[0][0] == min_vertical_advance_one_step]

        self.opp_action = random.choice(min_actions)


def row_distance(board: Board, player: int) -> float:
    '''
    For the current player and the chessboard, 
    we calculate a value for each of the player's chess pieces, 
    and the value is larger when the piece is closer to its target position.
    We add all the values up as "sum_1".
    We do the same thing for the opponent player to get "sum_2"
    sum_1 - sum_2 can be taken to evaluate the quality of the state for the player. 
    '''
    status = board.board_status
    res = 0
    if player == 1:
        for pos, value in status.items():
            row = pos[0]
            if value == 1:
                if row == 1:
                    res += 350
                elif row == 3:
                    res += 300
                elif row == 4:
                    res += 250
                elif row == 2:
                    res += 200
                else:
                    res += (20 - row)
            elif value == 3:
                if row == 2:
                    res += 1000
                elif row == 1 or row == 3:
                    res += 400
                elif row == 4:
                    res += 250
                else:
                    res += (20 - row) * 1.5
            elif value == 2:
                if row == 19:
                    res -= 350
                elif row == 17:
                    res -= 300
                elif row == 16:
                    res -= 250
                elif row == 18:
                    res -= 200
                else:
                    res -= row
            elif value == 4:
                if row == 18:
                    res -= 1000
                elif row == 19 or row == 17:
                    res -= 400
                elif row == 16:
                    res -= 250
                else:
                    res -= row * 1.5
    
    else:
        for pos, value in status.items():
            row = pos[0]
            if value == 1:
                if row == 1:
                    res -= 350
                elif row == 3:
                    res -= 300
                elif row == 4:
                    res -= 250
                elif row == 2:
                    res -= 200
                else:
                    res -= (20 - row)
            elif value == 3:
                if row == 2:
                    res -= 1000
                elif row == 1 or row == 3:
                    res -= 400
                elif row == 4:
                    res -= 250
                else:
                    res -= (20 - row) * 1.5
            elif value == 2:
                if row == 19:
                    res += 350
                elif row == 17:
                    res += 300
                elif row == 16:
                    res += 250
                elif row == 18:
                    res += 200
                else:
                    res += row
            elif value == 4:
                if row == 18:
                    res += 1000
                elif row == 19 or row == 17:
                    res += 400
                elif row == 16:
                    res += 250
                else:
                    res += row * 1.5
    
    return res

def concentrate(board: Board, player: int) -> float:
    """
    Calculate the concentration degree(集中程度) of the player's chess pieces as "res_1".
    We do the same thing for the opponent player to get "res_2".
    sum_1 - sum_2 can be taken to evaluate the quality of the state for the player
    The more concentrated the chess pieces are, the possibility of Continuous jumping for the player is higher.
    """
    status = board.board_status
    list_1, list_2 = [], []
    for pos, value in status.items():
        if value == 1 or value == 3:
            list_1.append(pos)
        elif value == 2 or value == 4:
            list_2.append(pos)
    
    res1, res2 = 0,0
    for i in range(10):
        for j in range(i+1,10):
            res1 += (abs(list_1[i][0]-list_1[j][0]) + abs(list_1[i][1]-list_1[j][1]))
            res2 += (abs(list_2[i][0]-list_2[j][0]) + abs(list_2[i][1]-list_2[j][1]))

    if player == 1:
        return res2 - res1
    else:
        return res1 - res2
    

def get_arrived_num(board, player):
    """
    get the number of arrived cheese pieces for the player.
    "noraml_num" is the number of arrived normal cheese pieces.
    "specail_num" is the number of arrived specail cheese pieces.
    """
    status = board.board_status
    noraml_num, special_num = 0,0
    if player == 1:
        for pos, value in status.items():
            if pos[0] <= 4 and pos[0] != 2 and value == 1:
                noraml_num += 1
            if pos[0] == 2 and value == 3:
                special_num += 1
            
    else:
        for pos, value in status.items():
            if pos[0] >= 16 and pos[0] != 18 and value == 2:
                noraml_num += 1
            if pos[0] == 18 and value == 4:
                special_num += 1
    return noraml_num, special_num

def first_last_distance(board: Board, player):
    """
    Find one player's cheese piece whose distance to its target position is farest among all the pieces.
    Return the farest distance.
    """
    status = board.board_status    
    if player == 1:
        last_row = 0
        for pos, value in status.items():
            if value == 1 or value == 3 and pos[0] > last_row:
                last_row = pos[0]
        return abs(last_row - 5)

    else:
        last_row = 20
        for pos, value in status.items():
            if value == 2 or value == 4 and pos[0] < last_row:
                last_row = pos[0]
        return abs(last_row - 15)

def evaluate_func(state) -> float:
    """
    Consider some factors(the distance of cheese pieces to their target position, 
    the concentration degree of the chess pieces(optional),
    and the farest distance of the cheese pieces to the target position),
    to evalaute the qaulity of the current state for the current player.
    """
    player: int = state[0]
    player = 3 - player
    board: Board = state[1]
    status = board.board_status

    (normal_num, special_num) = get_arrived_num(board, player)

    tem_1 = row_distance(board, player)
    tem_2 = concentrate(board, player)
    tem_3 = first_last_distance(board, player)

    if normal_num == 8 and special_num == 2:
        return 100000
    if normal_num == 7 and special_num == 2:
        init_score = 50000
        if player == 1:
            for pos, value in status.items():
                if value == 1 and pos[0] > 4:
                    init_score -= (pos[0] - 4) * 10
        else:
            for pos, value in status.items():
                if value == 2 and pos[0] < 16:
                    init_score -= (16 - pos[0]) * 10
        return init_score
    
    # score = 10 * tem_1 + 0.02 * tem_2 - 20 * tem_3
    score = tem_1 - 20 * tem_3
    # score = tem_1
    return score


def one_move(board: Board, action) -> Board:
    """
    get the new board after one move "action" based on the original board. 
    """
    pos_1 = action[0]
    pos_2 = action[1]
    new_board = copy.deepcopy(board)
    cheese = new_board.board_status[pos_1]
    new_board.board_status[pos_1] = 0
    new_board.board_status[pos_2] = cheese
    return new_board


def init_judge(action, player, type) -> bool:
    """
    Forcefully negate some actions(The function values for these actions are False):
    1. make the chess pieces that have not moved to the target position move backwards
    2. move the special chess piece that has been moved to the target position
    3. move the normal chess piece that has been moved to the innermost position
    """
    pos_1 = action[0]
    pos_2 = action[1]
    if player == 1:
        if pos_2[0] >= pos_1[0] and pos_1[0] >= 5:
            return False
        if type == 1 and pos_1[0] == 1 and pos_2[0] > 1:
            return False
        if type == 3 and pos_1[0] == 2 and pos_2[0] != 2:
            return False

    else:
        if pos_1[0] <= 14 and pos_2[0] <= pos_1[0]:
            return False
        if type == 2 and pos_1[0] == 19 and pos_2[0] < 19:
            return False
        if type == 4 and pos_1[0] == 18 and pos_2[0] != 18:
            return False
        
    return True

def one_step_to_victory(board: Board, player):
    """
    judge whether one more step is enough to win for player
    """
    action_list = []
    player_piece_pos_list = board.getPlayerPiecePositions(player)
    for pos in player_piece_pos_list:
        for adj_pos in board.adjacentPositions(pos):
            if board.isEmptyPosition(adj_pos):
                action_list.append((pos, adj_pos))

    for pos in player_piece_pos_list:
        boardCopy = copy.deepcopy(board)
        boardCopy.board_status[pos] = 0
        for new_pos in boardCopy.getAllHopPositions(pos):
            if (pos, new_pos) not in action_list:
                action_list.append((pos, new_pos))
    for action in action_list:
        new_board = one_move(board, action)
        if get_arrived_num(new_board, player) == (8,2):
            return True
    
    return False

def evaluate_step(state, action) -> float:
    """
    get a score for one action for the current state
    Overall, the farther the action move the cheese piece or move the chess piece to the target position, 
    the higher the score for this action.
    If the action only move the cheese to its neighbor position, some score will be deducted.
    """
    score = 0
    player = state[0]
    board: Board = state[1]    
    pos_1 = action[0]
    pos_2 = action[1]
    type = board.board_status[pos_1]

    if player == 1:
        if pos_1[0] >= 6 and pos_2 in board.adjacentPositions(pos_1):
            score -= 200
        if pos_1[0] >= 6 and pos_2[0] < pos_1[0]:
            score += (pos_1[0] - pos_2[0]) * 10
        if type == 1 and pos_1[0] > 4 and pos_1[0] <= 4:
            if pos_1[0] != 2:
                score += 50
            else:
                score += 20
        if type == 3 and pos_1[0] != 2 and pos_2[0] == 2:
            score += 200
        if type == 3 and pos_1[0] == 2 and pos_2[0] != 2:
            score -= 1000
        if type == 1 and pos_1[0] >= 16 and pos_2[0] < 16:
            score += 100
        if type == 3 and pos_1[0] >= 16 and pos_2[0] < 16:
            score += 200  

    else:
        if pos_1[0] <= 14 and pos_2 in board.adjacentPositions(pos_1):
            score -= 200
        if pos_1[0] <= 14 and pos_2[0] > pos_1[0]:
            score += (pos_2[0] - pos_1[0]) * 10
        if type == 2 and pos_1[0] < 16 and pos_1[0] <= 16:
            if pos_1[0] != 18:
                score += 50
            else:
                score += 20
        if type == 4 and pos_1[0] != 18 and pos_2[0] == 18:
            score += 200
        if type == 4 and pos_1[0] == 18 and pos_2[0] != 18:
            score -= 1000
        if type == 2 and pos_1[0] <= 4 and pos_2[0] > 4:
            score += 100
        if type == 4 and pos_1[0] <= 4 and pos_2[0] > 4:
            score += 200            

    return score


class YourAgent(Agent):
    def alpha_beta(self, state, depth, alpha, beta, maximizing_player):
        """
        Run minimax algorithm and do alpha_beta pruning.
        Only the actions that pass the initial verification and with scores in the top 30 will be selected each time.
        """
        player = self.game.player(state)
        board: Board = state[1]
        status = board.board_status
        if self.game.isEnd(state,0) or depth == 0:
            return evaluate_func(state), None

        init_actions = self.game.actions(state)
        second_actions = []
        for action in init_actions:
            new_board = one_move(board,action)
            if get_arrived_num(new_board, player) == (8,2):
                return evaluate_func(state), action
            
            type = status[action[0]]

            if init_judge(action,player,type):
                second_actions.append(action)
        action_list = []
        score_list = []
        for action in second_actions:
            step_score = evaluate_step(state, action)
            score_list.append(step_score)

        indexed_scores = list(enumerate(score_list))
        sorted_indexed_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
        sorted_indices = [index for index, score in sorted_indexed_scores]

        for i in range(len(sorted_indices)):
            if i < 30:
                action_list.append(second_actions[i])
        
        best_action = None

        if maximizing_player:
            max_eval = float('-inf')
            for action in action_list:
                eval, _ = self.alpha_beta(self.game.succ(state, action), depth - 1, alpha, beta, False)
                # print(action, eval)
                if eval > max_eval:
                    max_eval = eval
                    best_action = action
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_action
        else:
            min_eval = float('inf')
            for action in action_list:
                eval, _ = self.alpha_beta(self.game.succ(state, action), depth - 1, alpha, beta, True)
                if eval < min_eval:
                    min_eval = eval
                    best_action = action
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_action
        
    def alpha_beta_1(self, state, depth, alpha, beta, maximizing_player):
        """
        Based on the minimax algorithm and alpha-beta pruning, 
        we demand that the action can not be repeated with the last 10 times.
        """
        history_action = self.action_list[-10:]

        player = self.game.player(state)
        board: Board = state[1]
        status = board.board_status
        if self.game.isEnd(state,0) or depth == 0:
            return evaluate_func(state), None

        init_actions = self.game.actions(state)
        second_actions = []
        for action in init_actions:
            new_board = one_move(board,action)
            if get_arrived_num(new_board, player) == (8,2):
                return evaluate_func(state), action
            
            type = status[action[0]]

            if (init_judge(action,player,type)
                and (action not in history_action)
                and (action[::-1] not in history_action)):
                second_actions.append(action)

        action_list = []
        score_list = []
        for action in second_actions:
            step_score = evaluate_step(state, action)
            score_list.append(step_score)

        indexed_scores = list(enumerate(score_list))
        sorted_indexed_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
        sorted_indices = [index for index, score in sorted_indexed_scores]

        for i in range(len(sorted_indices)):
            if i < 30:
                action_list.append(second_actions[i])
        
        best_action = None

        if maximizing_player:
            max_eval = float('-inf')
            for action in action_list:
                eval, _ = self.alpha_beta(self.game.succ(state, action), depth - 1, alpha, beta, False)
                if eval > max_eval:
                    max_eval = eval
                    best_action = action
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_action
        else:
            min_eval = float('inf')
            for action in action_list:
                eval, _ = self.alpha_beta(self.game.succ(state, action), depth - 1, alpha, beta, True)
                if eval < min_eval:
                    min_eval = eval
                    best_action = action
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_action


    def getAction(self, state):
        """
        run minimax algorithm with depth = 3, not considering the history actions.
        When finding the output action is opposite to the last action,
        run minimax algorithm with depth = 1, considering the history actions.
        (By testing, we find that after the cheese piece start looping, 
        the performance of minimax with depth=1 is a little better than that with depth=3.)
        """
        player = state[0]
        board: Board = state[1]
        if self.depth == 3:  
            res = self.alpha_beta(state, depth = 3, alpha=-float('inf'), beta=float('inf'), maximizing_player= True)
        else:
            res = self.alpha_beta_1(state, depth = 1, alpha=-float('inf'), beta=float('inf'), maximizing_player= True)
        self.action = res[1]
        
        if len(self.action_list) > 10 and self.action[::-1] == self.action_list[-1]:
            self.depth = 1
            new_res = self.alpha_beta_1(state, depth = 1, alpha=-float('inf'), beta=float('inf'), maximizing_player= True)
            self.action = new_res[1]

        self.action_list.append(self.action)
        self.iter += 2
        print(self.action_list)
        
        new_board = one_move(board, self.action)
        if get_arrived_num(new_board, player) == (8,2):
            self.depth = 3
        if one_step_to_victory(new_board, 3-player):
            self.depth = 3
        if self.iter >= self.max_iteration:
            self.depth = 3
            self.iter = 0

        

    def oppAction(self, state):
        res = self.alpha_beta(state, depth = 3, alpha=-float('inf'), beta=float('inf'), maximizing_player= False)            
        self.opp_action = res[1]