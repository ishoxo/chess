import chess
sx = ['a', 'b', 'c', 'd', 'e', 'd', 'f', 'g', 'h']
sy = ['1', '2', '3', '4', '5', '6', '7', '8']
def squares_from(x,y):
    moves = []
    moev_name = []
    uu = min((8-x), (8-y))
    dd = min((x-1), (y-1))
    ud = min((8-x), (y-1))
    du = min((x-1), (8-y))
    l = x
    r = 8 - x
    d = y
    u = 8 - y
    for i in range(0, uu):
        moves.append(sx[x+i] + sy[y+i])
    #for i in range(0, dd):
        #moves.append(sx[x-i] + sy[y-i])
    #for i in range(0, ud):
        #moves.append(sx[x + i] + sy[y-i])
    #for i in range(0, du):
        #print('i:', i)
        #moves.append(sx[x-i] + sy[y+1])
    #for i in range(0, l):
        #moves.append(sx[x-i] + sy[y])
    #for i in range(0, r):
        #moves.append(sx[x+i] + sy[y])
    #for i in range(0, d):
        #moves.append(sx[x] + sy[y-i])
    #for i in range(0, u):
        #moves.append(sx[x] + sy[y+i])

    return moves


a = squares_from(2,2)
print(a)
print(len(a))

print(sx[7])

