import os
import sys
fdir = os.path.dirname(__file__)
sys.path.insert(0, './{}/../src/'.format(fdir))
import grid_world as gw

'''
Make maps and ground truth files
'''
# rows = 100
# cols = 100
# for i in xrange(10):
#     fname = './maps/map{}'.format(i)
#     G = gw.generate_map(rows, cols, in_file=fname, save_file=True)
#     for j in xrange(10):
#         gw.generate_ground_truth_data(G, map_num=i, ground_truth_num=j)

mn = 0
gtn = 1
G = gw.load_map_from_file('./maps/map{}'.format(mn))
C0, C, A, E = gw.show_ground_truth_data('./maps/map{}_groundtruth{}'.format(mn, gtn))
print(C0)
print(C)
print(A)
print(E)
gw.show_map(G)
