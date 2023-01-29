import numpy as np

from Board import BoardUtility
import random


class Player:
    def __init__(self, player_piece):
        self.piece = player_piece

    def play(self, board):
        return 0


class RandomPlayer(Player):
    def play(self, board):
        return random.choice(BoardUtility.get_valid_locations(board))


class HumanPlayer(Player):
    def play(self, board):
        move = int(input("input the next column index 0 to 8:"))
        return move


class MiniMaxPlayer(Player):
    def __init__(self, player_piece, depth=5):
        super().__init__(player_piece)
        self.depth = depth

    def play(self, board):
        """
        Inputs : 
           board : 7*9 numpy array. 0 for empty cell, 1 and 2 for cells containig a piece.
        return the next move(columns to play in) of the player based on minimax algorithm.
        """
        # Todo: implement minimax algorithm with alpha beta pruning
        def minimax(board, depth, alpha, beta, player):
            if depth == 0 or BoardUtility.is_terminal_state(board):
                return BoardUtility.score_position(board, self.piece), None
            moves = BoardUtility.get_valid_locations(board)
            opponent = 1 if player == 2 else 2
            is_max = True if player == self.piece else False
            best_score = -np.Inf if is_max else np.Inf
            best_move = 0
            for move in moves:
                clone = board.copy()
                BoardUtility.make_move(clone, move, player)
                score, _ = minimax(clone, depth - 1, alpha, beta, opponent)
                if is_max:
                    if score > best_score:
                        best_score = score
                        best_move = move
                    if score >= beta:
                        return score, move
                    alpha = max(alpha, score)
                else:
                    if score < best_score:
                        best_score = score
                        best_move = move
                    if score <= alpha:
                        return score, move
                    beta = min(beta, score)
            return best_score, best_move

        _, move = minimax(board, self.depth, -np.inf, np.inf, self.piece)
        return move
    
class MiniMaxProbPlayer(Player):
    def __init__(self, player_piece, depth=5, prob_stochastic=0.1):
        super().__init__(player_piece)
        self.depth = depth
        self.prob_stochastic = prob_stochastic


    def play(self, board):
        """
        Inputs : 
           board : 7*9 numpy array. 0 for empty cell, 1 and 2 for cells containig a piece.
        same as above but each time you are playing as max choose a random move instead of the best move
        with probability self.prob_stochastic.
        """
        # Todo: implement minimax algorithm with alpha beta pruning
        def minimax(board, depth, alpha, beta, player):
            if depth == 0 or BoardUtility.is_terminal_state(board):
                return BoardUtility.score_position(board, self.piece), None
            moves = BoardUtility.get_valid_locations(board)
            opponent = 1 if player == 2 else 2
            is_max = True if player == self.piece else False
            best_score = -np.Inf if is_max else np.Inf
            best_move = 0
            for move in moves:
                clone = board.copy()
                BoardUtility.make_move(clone, move, player)
                score, _ = minimax(clone, depth - 1, alpha, beta, opponent)
                if is_max:
                    if score > best_score:
                        best_score = score
                        best_move = move
                    if score >= beta:
                        return score, move
                    alpha = max(alpha, score)
                else:
                    if score < best_score:
                        best_score = score
                        best_move = move
                    if score <= alpha:
                        return score, move
                    beta = min(beta, score)
            return best_score, best_move

        _, move = minimax(board, self.depth, -np.inf, np.inf, self.piece)
        if random.random() < self.prob_stochastic:
            return random.choice(BoardUtility.get_valid_locations(board))
        return move

