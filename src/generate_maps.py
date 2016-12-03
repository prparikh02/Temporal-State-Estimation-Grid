# Parth Parikh & William Grant
# CS520 - Assignment 3 - Programming Component
import matplotlib.pyplot as plt
import numpy as np


def generate_map(rows, cols, file=None, start=None, goal=None):
    '''
    TODO: implement *args
    Map Cell Legend --
    0: (N) normal
    1: (T) hard-to-traverse
    2: (H) highway
    3: (B) blocked
    '''
    G = np.zeros((rows, cols))
    add_hard_to_traverse_cells(G, nc=8, r=15)
    add_highways(G)
    add_blocked_cells(G)

    R, C = np.meshgrid(np.arange(rows+1), np.arange(cols+1))
    plt.pcolormesh(R, C, G, cmap='YlOrRd')
    plt.colorbar()
    plt.show()


def add_hard_to_traverse_cells(G, nc=8, r=10, p=0.50):
    '''
    G: 2d-array representing map/graph
    nc: number of centroids
    r: centroid radius
    p: probability of making hard-to-traverse cell in centroid radius
    '''
    rows, cols = G.shape
    centroids = [
        (np.random.randint(0, cols),
         np.random.randint(0, rows)) for i in xrange(nc)
    ]

    for center in centroids:
        for i in xrange(center[0]-r, center[0]+r+1):
            if i not in xrange(0, cols):
                continue
            for j in xrange(center[1]-r, center[1]+r+1):
                if j not in xrange(0, rows):
                    continue
                if np.random.uniform(0, 1) >= 0.50:
                    G[i, j] = 1


def add_highways(G, nh=4, min_length=100, seg_length=20):
    '''
    G: 2d-array representing map/graph
    nh: number of highways
    min_length: minimum length of any highway
    seg_length: length of each segment

    Highways are created by appending a sequence of segments that pivot
    '''

    def create_highway(G, min_length, seg_length):
        r, c, direction = generate_highway_start_point(G.shape)
        if G[r, c] == 2:
            return set()
        highway = set()
        highway.add((r, c))  # add starting point to highway
        steps = 1

        while True:
            for i in xrange(seg_length):
                r, c = get_next_highway_point(r, c, direction)
                if G[r, c] == 2 or (r, c) in highway:
                    return set()
                if is_boundary_node(G.shape, r, c):
                    if steps >= min_length:
                        highway.add((r, c))
                        return highway
                    return set()
                highway.add((r, c))
                steps += 1
            direction = choose_next_direction(direction)

    def generate_highway_start_point(dims):
        '''
        dims: (tup) dimensions of map/graph

        Returns (start_r, start_c, direction_to_move)
        direction_to_move is char in ['n', 's', 'e', 'w']
        '''
        rows, cols = dims
        # starting point for the highway should be at boundary,
        #   but not at the corners
        boundary = np.random.choice(['n', 's', 'e', 'w'])
        if boundary == 'n':
            return (0, np.random.randint(1, cols-1), 's')
        elif boundary == 's':
            return (rows-1, np.random.randint(1, cols-1), 'n')
        elif boundary == 'e':
            return (np.random.randint(1, rows-1), cols-1, 'w')
        else:
            return (np.random.randint(1, rows-1), 0, 'e')

    def get_next_highway_point(r, c, direction):
        if direction == 'n':
            return (r-1, c)
        elif direction == 's':
            return (r+1, c)
        elif direction == 'e':
            return (r, c+1)
        else:
            return (r, c-1)


    def is_boundary_node(dims, r, c):
        rows, cols = dims
        return (r in [0, rows-1] or
                c in [0, cols-1])


    def choose_next_direction(curr_direction):
        if curr_direction in ['n', 's']:
            perp_directions = ['e', 'w']
        else:
            perp_directions = ['n', 's']

        if np.random.uniform(0, 1) < 0.60:
            return curr_direction
        return np.random.choice(perp_directions)

    for i in xrange(nh):
        highway = set()
        while not highway:
            highway = create_highway(G, min_length, seg_length)

        for r, c in highway:
            G[r, c] = 2


def add_blocked_cells(G, p=0.20):
    '''
    G: 2d-array representing map/graph
    p: percentage of cells to block
    '''
    N = G.size
    B = int(N*p)

    C = np.arange(N)
    while B != 0:
        n = np.random.choice(C)
        if G.ravel()[n] == 2:
            continue
        G.ravel()[n] = 3
        B -= 1
