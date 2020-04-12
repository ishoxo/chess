import chess
import chess.svg
import IPython
from IPython.display import SVG
board = chess.Board()
SVG(chess.svg.board(board=board, size=400))
print(board)

print(board)
print(chess.E4)
#%%

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
NUM = [1,2,3,4,5,6,7,8]
p = ['', 'N', 'B', 'R', 'Q', 'K']

squares = []
for i in range(len(letters)):
    for j in range(len(NUM)):
        s = letters[i] + str(NUM[j])
        squares.append(s)
print(squares)

moves = []

for i in squares:
    for j in squares:
        if i != j:
            n = i + j
            moves.append(n)

move_no = []
for i in range(len(moves)):
    move_no.append(i)


move_dictionary= {}
for key in moves:
    for value in move_no:
        move_dictionary[key] = value
        move_no.remove(value)
        break

print(move_dictionary)
print(len(move_dictionary))

#%%
move_dictionary['a1a2']







#%%


def board_to_vec(board):
    pieces_dict = {'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6, 'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6}
    print(chess.E4)
    A1 = chess.A1
    A2 = chess.A2
    A3 = chess.A3
    A4 = chess.A4
    A5 = chess.A5
    A6 = chess.A6
    A7 = chess.A7
    A8 = chess.A8
    B1 = chess.B1
    B2 = chess.B2
    B3 = chess.B3
    B4 = chess.B4
    B5 = chess.B5
    B6 = chess.B6
    B7 = chess.B7
    B8 = chess.B8
    C1 = chess.C1
    C2 = chess.C2
    C3 = chess.C3
    C4 = chess.C4
    C5 = chess.C5
    C6 = chess.C6
    C7 = chess.C7
    C8 = chess.C8
    D1 = chess.D1
    D2 = chess.D2
    D3 = chess.D3
    D4 = chess.D4
    D5 = chess.D5
    D6 = chess.D6
    D7 = chess.D7
    D8 = chess.D8
    E1 = chess.E1
    E2 = chess.E2
    E3 = chess.E3
    E4 = chess.E4
    E5 = chess.E5
    E6 = chess.E6
    E7 = chess.E7
    E8 = chess.E8
    F1 = chess.F1
    F2 = chess.F2
    F3 = chess.F3
    F4 = chess.F4
    F5 = chess.F5
    F6 = chess.F6
    F7 = chess.F7
    F8 = chess.F8
    G1 = chess.G1
    G2 = chess.G2
    G3 = chess.G3
    G4 = chess.G4
    G5 = chess.G5
    G6 = chess.G6
    G7 = chess.G7
    G8 = chess.G8
    H1 = chess.H1
    H2 = chess.H2
    H3 = chess.H3
    H4 = chess.H4
    H5 = chess.H5
    H6 = chess.H6
    H7 = chess.H7
    H8 = chess.H8
    squares = [A1, A2, A3, A4, A5, A6, A7, A8, B1, B2, B3, B4, B5, B6, B7, B8, C1, C2, C3, C4, C5, C6, C7, C8, D1, D2,
               D3, D4, D5, D6, D7, D8, E1, E2, E3, E4, E5, E6, E7, E8, F1, F2, F3, F4, F5, F6, F7, F8, G1, G2, G3, G4,
               G5, G6, G7, G8, H1, H2, H3, H4, H5, H6, H7, H8]
    board_vec = []
    for i in range(len(squares)):
        s = squares[i]
        if board.piece_at(s) is None:
            board_vec.append(0)
        else:
            board_vec.append(pieces_dict[board.piece_at(s).symbol()])
    return board_vec

#%%
print(board)
print(board_to_vec(board))
board.push_san("e4")
print(board_to_vec(board))
#%%


initial_board_state = [4, 1, 0, 0, 0, 0, -1, -4, 2, 1, 0, 0, 0, 0, -1, -2, 3, 1, 0, 0, 0, 0, -1, -3, 5, 1, 0, 0, 0, 0, -1, -5, 6, 1, 0, 0, 0, 0, -1, -6, 3, 1, 0, 0, 0, 0, -1, -3, 2, 1, 0, 0, 0, 0, -1, -2, 4, 1, 0, 0, 0, 0, -1, -4]
