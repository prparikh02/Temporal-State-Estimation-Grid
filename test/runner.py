import os
import sys
fdir = os.path.dirname(__file__)
sys.path.insert(0, './{}/../src/'.format(fdir))
import grid_world as gw
import numpy as np

'''
Make maps and ground truth files
'''
# rows = 100
# cols = 100
# for i in xrange(10):
#     fname = './maps/map{}'.format(i)
#     G = gw.generate_map(rows, cols, in_file=fname, save_file=True)
#     for j in xrange(10):
#         print (i, j)
#         C0, C, A, E = gw.generate_ground_truth_data(G)
#         GT = (C0, C, A, E)
#         gw.save_ground_truth(GT, i, j)


N = 1
for i in xrange(N):

    '''
    Random
    '''
    # N = 100
    # G = gw.generate_map(N, N)
    # GT = gw.generate_ground_truth_data(G)
    # ML = gw.filtering(G, GT)


    '''
    Loading from file
    '''
    mn = 3
    gtn = 2
    map_file = './maps/map{}'.format(mn)
    gt_file = './maps/map{}_groundtruth{}'.format(mn, gtn)
    G = gw.load_map_from_file(map_file)
    GT = gw.load_ground_truth_data(gt_file)
    results = gw.filtering(G, GT, show_heatmaps=True, show_error=True)
    gw.display_map(G)
    MLS, Err = gw.viterbi(G, GT, show_trajectories=True, show_error=True)
    

    '''
    # Assignment Example
    # '''
    # G = np.array([[2, 2, 1], [0, 0, 0], [0, 3, 2]])
    # A = ['R', 'R', 'D', 'D']
    # E = ['N', 'N', 'H', 'H']
    # GT = (None, None, A, E)
    # gw.filtering(G, GT)
    # MLS, Err = gw.viterbi(G, GT, N=8)
    # print(MLS)
    # print(Err)

    # C0, C, A, E = GT
    # print('initial: {}'.format(C0))
    # print('final: {}'.format(C[-1]))
    # print('Filtering estimate: {}'.format(ML))
    # print('Manhattan Error: {}'.format(abs(ML[0][0]-C[-1][0]) + abs(ML[0][1]-C[-1][1])))
