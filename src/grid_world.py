# Parth Parikh & William Grant
# CS520 - Assignment 3 - Programming Component
import os
import matplotlib.pyplot as plt
import numpy as np


def generate_map(rows, cols, in_file=None, start=None, save_file=False):
    '''
    Args --
    rows: number of rows in map
    cols: number of columns in map
    in_file: loads map from input file (disregards rows and columns)
    start: tuple of starting point (r, c)
    save_file: boolean flag to save map to file

    Map Cell Legend --
    0: (N) normal
    1: (T) hard-to-traverse
    2: (H) highway
    3: (B) blocked
    '''
    if in_file and os.path.isfile(in_file):
        return load_map_from_file(in_file)
    G = np.zeros((rows, cols))
    add_hard_to_traverse_cells(G, nc=8, r=15)
    add_highways(G)
    add_blocked_cells(G)
    G = G.astype(int)
    if save_file:
        i = 0
        while os.path.exists('./maps/map{}'.format(i)):
            i += 1
        np.savetxt('./maps/map{}'.format(i), G, fmt='%d', delimiter=',')
    return G


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

    # begin creating highways
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


def generate_start_point(G):
    '''
    G: 2d-array representing map/graph
    '''

    rows, cols = G.shape
    while True:
        rs = np.random.randint(rows)
        cs = np.random.randint(cols)
        if G[rs, cs] != 3:
            return (rs, cs)


def generate_ground_truth_data(G, sp=None, **kwargs):
    '''
    G: 2d-array representing map/graph
    TODO: implement *args
    '''

    if 'sp' not in kwargs:
        r, c = generate_start_point(G)
    else:
        r, c = kwargs['sp']

    N = 100
    actions = ['U', 'L', 'D', 'R']
    readings = ['N', 'H', 'T']
    # generate N random actions
    A = [np.random.choice(actions) for i in xrange(N)]
    # sequence of coordinates
    C0 = to_xy(r, c)
    C = [(-1, -1) for i in xrange(N)]
    # sequence of sensor readings
    E = ['' for i in xrange(N)]

    # set transition and observational model probabilities
    tm = 0.90
    om = 0.90
    for i in xrange(N):
        if np.random.uniform() < tm:
            rp, cp = action_coords(A[i], r, c)  # potential new location
            if is_occupiable(G, rp, cp):
                r, c = rp, cp
        C[i] = to_xy(r, c)
        true_sr = sensor_reading(G[C[i]])  # true sensor reading
        if np.random.uniform() < om:
            sr = true_sr
        else:
            sr = np.random.choice([sr for sr in readings if sr not in true_sr])
        E[i] = sr
    
    if 'map_num' not in kwargs and 'ground_truth_num' not in kwargs:
        print(C0)
        print(C)
        print(A)
        print(E)
        return
    
    mn = kwargs['map_num']
    gtn = kwargs['ground_truth_num']
    fname = './maps/map{}_groundtruth{}'.format(mn, gtn)
    with open(fname, 'w') as f:
        f.write(str(C0) + '\n')
        for c in C:
            f.write(str(c) + '\n')
        for a in A:
            f.write(a + '\n')
        for e in E:
            f.write(e + '\n')


def show_ground_truth_data(fname):
    with open(fname, 'r') as f:
        data = f.readlines()
    data = map(lambda s: s.strip(), data)
    C0 = data[0]
    C = data[1:101]
    A = data[101:201]
    E = data[201:301]
    return (C0, C, A, E)


def sensor_reading(x):
    '''
    returns x -> e in ['N', 'H', 'T', 'B']
    '''

    E = ['N', 'H', 'T', 'B']
    return E[x]


def is_occupiable(G, r, c):
    '''
    G: 2d-array representing map/graph
    r, c: current coordinates
    '''
    rows, cols = G.shape
    if r >= 0 and r < rows and c >= 0 and c < cols:
        return G[r, c] != 3
    return False


def action_coords(action, r, c):
    '''
    action: ['U', 'L', 'D', 'R']
    r, c: current coordinates

    returns new coordinates with action applied
    '''
    if action == 'U':
        return (r-1, c)
    elif action == 'D':
        return (r+1, c)
    elif action == 'R':
        return (r, c+1)
    else:
        return (r, c-1)


def to_xy(r, c):
    '''
    convert (row, column) to (x, y)
    '''
    return (c, r)


def to_rc(x, y):
    '''
    convert (x, y) to (row, column)
    '''
    return (y, x)

def load_map_from_file(fname):
    return np.loadtxt(fname, delimiter=',', dtype=int)


def show_map(G, *args):
    '''
    G: 2d-array representing map/graph
    TODO: implement *args
    '''

    rows, cols = G.shape
    cmap = 'YlOrRd'
    R, C = np.meshgrid(np.arange(rows+1), np.arange(cols+1))
    plt.pcolormesh(R, C, G, cmap=cmap)
    # plt.colorbar()
    plt.gca().invert_yaxis()
    plt.gca().xaxis.set_ticks_position('top')
    plt.show()


def show_heatmap(H, *args):
    '''
    H: 2d-array representing probability heatmap
    TODO: implement *args
    '''

    rows, cols = G.shape
    cmap = 'reds'
    R, C = np.meshgrid(np.arange(rows+1), np.arange(cols+1))
    plt.colormesh(R, C, H, cmap=cmap)
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.gca().xaxis.set_ticks_position('top')
    plt.show()
