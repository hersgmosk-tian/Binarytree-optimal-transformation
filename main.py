import math
from tqdm import tqdm
import time
from random import randint
import trans_ori
import trans_update

def random_nonleaf_seq(n):
    """随机生成 n 层的非叶节点序列"""
    assert n>0, "层数至少为1"
    if n==1: return [0]
    seq = [1]
    while len(seq)!=n-1:
        total = seq[-1] * 2
        seq.append(randint(1,total))
    return seq+[0]

def main():
    '''
    p = [-1, -2, -4, -6, -10, 11, 21, 35, -7, 40, -11, -5, -8, -9, -12, -14, -16, 12, 5, 50, -15, 20, -17, 111, 9, -13, -3]
    i = [11, -10, 21, -6, 35, -4, 40, -7, -11, -2, -8, -5, 12, -16, 5, -14, 50, -12, 20, -15, 111, -17, 9, -9, -13, -1, -3]
    arr = trans.buildTree(p, i)
    arr = trans.buildNums(arr, 8)
    n = 8
    positions = []
    for _ in range(8):
        tmp = arr.copy()
        positions.append(tmp)
    time_sum = 0
    nonleaf2leaf = lambda nonleaf:[0]+[2*a-b for a,b in zip(nonleaf[:-1],nonleaf[1:])]
    random_leaf_seq = lambda n: nonleaf2leaf(random_nonleaf_seq(n))
    for _ in range(50):
        tmp_tar = random_leaf_seq(8)
        for k in range(len(tmp_tar)):
            tmp_tar[k] *= 8
        time_start=time.time()
        transm.test(positions, tmp_tar)
        time_end = time.time()
        time_sum += (time_end - time_start)
        print(tmp_tar)
    time_sum /= 50
    print('totally cost',time_sum)
    
    
    '''

    time_sum_alg1 = 0
    time_sum_alg2 = 0
    nonleaf2leaf = lambda nonleaf:[0]+[2*a-b for a,b in zip(nonleaf[:-1],nonleaf[1:])]
    random_leaf_seq = lambda n: nonleaf2leaf(random_nonleaf_seq(n))

    max_time_alg1 = -math.inf
    max_time_alg2 = -math.inf
    min_time_alg1 = math.inf
    min_time_alg2 = math.inf
    ans_list1 = []
    ans_list2 = []
    depth = 15
    # for _ in range(1000):
    for _ in tqdm(range(1000), "Tree Transfer Test"):
        positons = []
        for tree_k in range(8):
            # if tree_k ==  0:
                # tmp_arr = BTree.random_binary_tree(depth, 1, 200)
                # tmp_arr = BTree.tree_to_positions(tmp_arr)
                # positons.append(tmp_arr)
            # else:
            tmp_arr = trans_update.BTree.random_binary_tree(randint(1, depth), 0, 200)
            tmp_arr = trans_update.BTree.tree_to_positions(tmp_arr)
            positons.append(tmp_arr)
        # print(len(positons))
        # tmp_arr = BTree.random_binary_tree(7, 1)
        # tmp_arr = BTree.tree_to_positions(tmp_arr)
        tmp_tar = [0] * (depth+1)
        "这里加总了八棵树的叶子节点数量之和"
        for _ in range(1):
            ttmp_tar = random_leaf_seq(depth + 1)
            for kk in range(len(tmp_tar)):
                tmp_tar[kk] += ttmp_tar[kk]
        # print(positons[0][:10], tmp_tar)
        time_start=time.time()
        "position是树的现有状态，每个位置上的值为value或none，tmp_tar为target"
        ans_list1.append(trans_ori.test([positons[0]], tmp_tar))
        time_end_alg1 = time.time()
        alg1_spend_time = time_end_alg1-time_start
        # ans_list2.append(transm_update.test(positons, tmp_tar))
        ans_list2.append(trans_update.main(positons[0], tmp_tar))
        time_end_alg2 = time.time()
        alg2_spend_time = time_end_alg2-time_end_alg1
        
        # time_end = time.time()
        # spend_time = time_end - time_start
        max_time_alg1 = max(max_time_alg1, alg1_spend_time)
        min_time_alg1 = min(min_time_alg1, alg1_spend_time)
        time_sum_alg1 += alg1_spend_time

        max_time_alg2 = max(max_time_alg2, alg2_spend_time)
        min_time_alg2 = min(min_time_alg2, alg2_spend_time)
        time_sum_alg2 += alg2_spend_time
        # print(tmp_tar)
    time_sum_alg1 /= 1000
    time_sum_alg2 /= 1000
    # print(ans_list1 == ans_list1)
    print('平均时长_alg1：', time_sum_alg1)
    print('平均时长_alg2：', time_sum_alg2)
    print('最长花费_alg1：', max_time_alg1)
    print('最长花费_alg2：', max_time_alg2)
    print('最短花费_alg1：', min_time_alg1)    
    print('最短花费_alg2：', min_time_alg2) 
    return

if __name__ == '__main__':
    main()
