# Parth Parikh & William Grant
# CS520 - Assignment 3 - Programming Component
from __future__ import division
import os
import matplotlib.pyplot as plt
import numpy as np
from ast import literal_eval as make_tuple

NORMAL = 0
HARD_T0_TRAVERSE = 1
HIGHWAY = 2
BLOCKED = 3
TRANSITION_SUCCESS = 0.90
OBSERVATION_TRUE = 0.90


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


def add_hard_to_traverse_cells(G, nc=10, r=15, p=0.50):
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
                    G[i, j] = HARD_T0_TRAVERSE


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
        if G[r, c] == HIGHWAY:
            return set()
        highway = set()
        highway.add((r, c))  # add starting point to highway
        steps = 1

        while True:
            for i in xrange(seg_length):
                r, c = get_next_highway_point(r, c, direction)
                if G[r, c] == HIGHWAY or (r, c) in highway:
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
        if G.ravel()[n] in (HIGHWAY, BLOCKED):
            continue
        G.ravel()[n] = BLOCKED
        B -= 1


def generate_start_point(G):
    '''
    G: 2d-array representing map/graph
    '''

    rows, cols = G.shape
    while True:
        rs = np.random.randint(rows)
        cs = np.random.randint(cols)
        if G[rs, cs] != BLOCKED:
            return (rs, cs)


def generate_ground_truth_data(G, **kwargs):
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
    C0 = (r, c)
    C = [(-1, -1) for i in xrange(N)]
    # sequence of sensor readings
    E = ['' for i in xrange(N)]

    # set transition and observational model probabilities
    tm = TRANSITION_SUCCESS
    om = OBSERVATION_TRUE
    for i in xrange(N):
        if np.random.uniform() < tm:
            rp, cp = action_coords(A[i], r, c)  # potential new location
            if is_occupiable(G, rp, cp):
                r, c = rp, cp
        C[i] = (r, c)
        true_sr = sensor_reading(G[C[i]])  # true sensor reading
        if np.random.uniform() < om:
            sr = true_sr
        else:
            sr = np.random.choice([sr for sr in readings if sr not in true_sr])
        E[i] = sr

    if 'map_num' in kwargs and 'ground_truth_num' in kwargs:
        GT = (C0, C, A, E)
        mn = kwargs['map_num']
        gtn = kwargs['ground_truth_num']
        save_ground_truth(GT, mn, gtn)

    return (C0, C, A, E)


def save_ground_truth(GT, mn, gtn):
    '''
    GT: (tuple) such that GT=(C0, C, A, E) where
        C0 is initial location,
        C is sequence of actual steps,
        A is sequence of actions,
        E is sequence of sensor readings
    mn: map number
    gtn: ground truth number for given map
    '''

    C0, C, A, E = GT

    fname = './maps/map{}_groundtruth{}'.format(mn, gtn)
    with open(fname, 'w') as f:
        f.write(str(C0) + '\n')
        for c in C:
            f.write(str(c) + '\n')
        for a in A:
            f.write(a + '\n')
        for e in E:
            f.write(e + '\n')


def load_ground_truth_data(fname):
    with open(fname, 'r') as f:
        data = f.readlines()
    data = map(lambda s: s.strip(), data)
    C0 = make_tuple(data[0])
    C = [make_tuple(d) for d in data[1:101]]
    A = data[101:201]
    E = data[201:]
    return (C0, C, A, E)


def filtering(G, GT, show_heatmaps=False):
    '''
    G: 2d-array representing map/graph
    GT: (tuple) such that GT=(C0, C, A, E) where
        C0 is initial location,
        C is sequence of actual steps,
        A is sequence of actions,
        E is sequence of sensor readings
    '''

    def get_possible_previous_states(r, c, a):
        '''
        given current state (r, c) and action a, return set of
        possible previous states that could have led to current state
        '''
        possible_states = [(r, c)]
        if a == 'U':
            rp, cp = (r+1, c)
        elif a == 'D':
            rp, cp = (r-1, c)
        elif a == 'R':
            rp, cp = (r, c-1)
        else:
            rp, cp = (r, c+1)

        if is_occupiable(G, rp, cp):
            possible_states.append((rp, cp))

        return possible_states

    rows, cols = G.shape
    C0, C, A, E = GT

    path_provided = False
    if C0 and C:
        path_provided = True
        # convert coordinates from xy to rc
        # C0 = [to_rc(C0[0], C0[1])]
        # C = [to_rc(c[0], c[1]) for c in C]
        C0 = [C0]

        # create running error list:
        #   For each tuple:
        #       1st: Actual Position, C[i]
        #       2nd: MLE position
        #       3rd: Manhattan error
        Err = [tuple() for i in xrange(len(C))]
        mle = [tuple() for i in xrange(len(C))]

    # create heatmap and set initial probabilities
    H = np.zeros(G.shape)
    H[G != BLOCKED] = 1/(G.size - (G == BLOCKED).sum())

    H_next = np.zeros(H.shape)
    for i in xrange(len(A)):
        a = A[i]
        e = E[i]
        for r in xrange(rows):
            for c in xrange(cols):
                if G[r, c] == BLOCKED:
                    H_next[r, c] == 0
                    continue
                # get possible previous states
                possible_prevs = get_possible_previous_states(r, c, a)

                # set transition model and prior belief probabilities
                if len(possible_prevs) == 1:
                    # if only one possible previous state, namely the current
                    #   state, then the transition model reduces to prior prob
                    rn, cn = action_coords(a, r, c)

                    trans_model = np.array((1.00, 0.00))
                    priors = np.array((H[r, c], 0.00))
                else:
                    rn, cn = action_coords(a, r, c)
                    trans_model = np.array((1.00, TRANSITION_SUCCESS))
                    priors = np.array((H[r, c], H[possible_prevs[1]]))

                # we need to do additional checks on the current state. Namely,
                #   check if the action could have even been successfully
                #   applied
                rn, cn = action_coords(a, r, c)
                if is_occupiable(G, rn, cn):
                    trans_model[0] = 1 - TRANSITION_SUCCESS

                # set observation modle probability
                if e == sensor_reading(G[r, c]):
                    obs_model = OBSERVATION_TRUE  # correct reading
                else:
                    obs_model = (1 - OBSERVATION_TRUE)/2.00

                H_next[r, c] = obs_model*trans_model.dot(priors)

        alpha = 1.0/sum(sum(H_next))
        H_next = alpha*H_next
        H = np.copy(H_next)

        mle[i] = np.unravel_index(H.argmax(), H.shape)
        Err[i] = (C[i], mle, np.absolute(np.array(C[i])-np.array(mle[i])).sum())

        if show_heatmaps:
            if i in (np.array([10, 50, 100])-1):
                if path_provided:
                    display_heatmap(H, C0 + C[:i])
                else:
                    display_heatmap(H)

    return (Err, H.max())


def sensor_reading(x):
    '''
    returns x -> e in ['N', 'T', 'H', 'B']
    '''

    E = ['N', 'T', 'H', 'B']
    return E[x]


def is_occupiable(G, r, c):
    '''
    G: 2d-array representing map/graph
    r, c: current coordinates
    '''
    rows, cols = G.shape
    if r >= 0 and r < rows and c >= 0 and c < cols:
        return G[r, c] != BLOCKED
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


def display_map(G, S=None, *args):
    '''
    G: 2d-array representing map/graph
    S: ground truth sequence of moves
    TODO: implement *args
    '''

    rows, cols = G.shape
    cmap = 'YlOrRd'
    R, C = np.meshgrid(np.arange(rows+1), np.arange(cols+1))
    plt.pcolormesh(R, C, G, cmap=cmap)
    # plt.colorbar()
    plt.gca().invert_yaxis()
    plt.gca().xaxis.set_ticks_position('top')
    if S:
        r = np.array([s[0] for s in S]) + 0.50
        c = np.array([s[1] for s in S]) + 0.50
        plt.plot(c, r)
        plt.plot(c[0], r[0], marker='o')
        plt.plot(c[-1], r[-1], marker='H')
    plt.show()


def display_heatmap(H, S=None, *args):
    '''
    H: 2d-array representing probability heatmap
    S: ground truth sequence of moves
    TODO: implement *args
    '''

    rows, cols = H.shape
    cmap = 'Reds'
    R, C = np.meshgrid(np.arange(rows+1), np.arange(cols+1))
    plt.pcolormesh(R, C, H, cmap=cmap)
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.gca().xaxis.set_ticks_position('top')
    if S:
        r = np.array([s[0] for s in S]) + 0.50
        c = np.array([s[1] for s in S]) + 0.50
        plt.plot(c, r)
        plt.plot(c[0], r[0], marker='o')
        plt.plot(c[-1], r[-1], marker='H')
    plt.grid()
    plt.show()


def plot_error(Err):

    fig, ax = plt.subplots(1)
    t = range(1, 101)
    ax.plot(t, Err, 'r--*')
    plt.show()