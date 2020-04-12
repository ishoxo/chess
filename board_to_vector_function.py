import chess
board = chess.Board()
pieces_dict = {'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6, 'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6}


def board_to_vec(board):
    board_vec = []
    for square in chess.SQUARES:
        if board.piece_at(square) is None:
            board_vec.append(0)
        else:
            board_vec.append(pieces_dict[board.piece_at(square).symbol()])
    return board_vec


#%%
print(board)
print(board_to_vec(board))
