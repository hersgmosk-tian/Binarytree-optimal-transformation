from itertools import combinations
from binarytree import Node
from copy import deepcopy
from random import randint,sample
from math import factorial

class BTree(Node):
    """二叉树，继承 Node 的显示功能
    根节点位置记为1，深度记为0
    """
    def __init__(self, value = 0):
        """更新旧的初始化"""
        Node.__init__(self, value) # 使用原初始化
        # 新增属性
        self.depth = 0 # 根节点深度
        self.position = 1 # 序数表述
        self._max_depth = 0 # 总深度
        
    def new_tree_by_positions(self,positions):
        """最后一层按位置展开 -> 新树"""
        tree = deepcopy(self) # 复制对象
        for node in tree.last_layer: # 最后一层
            if node.position in positions:
                self.add_left_right_to_node(node) # 展开
        tree._max_depth += 1 # 总深度+1
        return tree
    
    @property
    def max_depth(self):
        """树的最长深度"""
        return self._max_depth
    
    @property
    def last_layer(self):
        """返回最后一层节点"""
        return [l for l in self.leaves if l.depth==self._max_depth]
    
    @classmethod
    def add_left_right_to_node(cls,node) -> None:
        """给 node 添加左右子结点"""
        cls.add_left_to_node(node) # 左节点
        cls.add_right_to_node(node) # 右节点
    
    @staticmethod
    def add_left_to_node(node,value=0) -> None:
        """给 node 添加左节点，设置了 depth 和 position 属性"""
        left = Node(value)
        left.depth = node.depth + 1
        left.position = 2 * node.position
        node.left = left
    
    @staticmethod
    def add_right_to_node(node,value=0) -> None:
        """给 node 添加右节点，设置了 depth 和 position 属性"""
        right = Node(value)
        right.depth = node.depth + 1
        right.position = 2 * node.position + 1
        node.right =  right
    
    @property
    def position_tree(self):
        """返回相同形状的树，结点显示值为位置"""
        tree = deepcopy(self)
        for node in tree:
            node.value = node.position
        return tree
    
    @classmethod
    def list_to_tree(cls,positions):
        """一维列表 -> 树，空节点用 None 表示"""
        n = len(positions)
        assert n, "输入列表不能为空"
        assert positions[0] is not None,"根节点不能为空"
        tree = BTree(positions[0]) # 初始化
        while True:
            last_layer = tree.last_layer
            flag = False # 标记是否有新节点
            for node in last_layer: # 检查是否有叶节点
                pos = 2 * node.position
                # 左节点
                if pos>n: continue
                if positions[pos-1] is not None:
                    flag = True
                    cls.add_left_to_node(node,positions[pos-1])
                # 右节点
                if pos+1>n: continue
                if positions[pos] is not None:
                    flag = True
                    cls.add_right_to_node(node,positions[pos])
            if flag: # 有新节点生成
                tree._max_depth += 1
            else:
                break
        return tree
    
    @staticmethod
    def random_binary_tree(depth, min_leaf = 0, max_value = 30):
        """生成随机树，叶子取值范围为 [0,max_value]"""
        t = BTree(0) # 初始树
        for i in range(depth):
            last_layer = t.last_layer # 最后一层
            if i == 0:
                split_num = randint(1,len(last_layer))
            else:
                # print(i)
                # print(len(last_layer), min_leaf)
                split_num = randint(1,len(last_layer) - min_leaf) # 随机数目
            split_leaves = sample(last_layer,split_num) # 随机位置
            for leaf in split_leaves:
                BTree.add_left_right_to_node(leaf)
            t._max_depth += 1 # 深度增加
        for leaf in t.leaves: # 叶子节点随机值
            leaf.value = randint(0,max_value)
        return t
    
    @staticmethod
    def tree_to_positions(tree):
        """树转列表"""
        depth = 16 # 树深度
        positions = [None for i in range(2**depth-1)]
        for node in tree:
            positions[node.position-1]=node.value
        return positions

    
def choose(data,n):
    """从 data 中取 n 个元素"""
    assert n>0,"不能取0个"
    if n > len(data) : return []
    res = [list(i) for i in combinations(data,n)]
    return res


def get_nodes_cost(tree):
    """获取节点的开销信息，只记录非0节点"""
    nonleaf_cost = {} # 非叶开销
    leaf_cost = {} # 叶节点开销
    for node in tree:
        if node.left is None: # 叶节点
            if node.value: 
                leaf_cost[node.position] = node.value
        else: # 非叶节点
            v = sum(leaf.value for leaf in node.leaves)
            if v: nonleaf_cost[node.position] = v
    return nonleaf_cost,leaf_cost

### 主函数打包 ###
def main(positions,leaves):
    """函数打包"""
    n = len(leaves)
    old = BTree.list_to_tree(positions)
    nonleaf_cost,leaf_cost = get_nodes_cost(old) # 节点开销信息
    nonleaves = leaves_to_nonleaves(leaves)
    root = BTree(0)
    root.cost = 0 # 初始开销
    global min_cost
    min_cost = sum(leaf_cost.values()) # 最小开销
    optimal = None # 最优解
    tmp_trees = [root] # 待遍历集合
    while len(tmp_trees):
        tree = tmp_trees[0] # 取最小开销树
        tmp_trees = tmp_trees[1:]
        news,is_end = next_level(tree,nonleaves,nonleaf_cost,leaf_cost)
        # 处理新树
        if is_end: # 树已完全展开
            for new in news:
                if new.cost < min_cost:
                    min_cost = new.cost
                    optimal = new
                    tmp_trees = [tree for tree in tmp_trees if tree.cost<= min_cost]
        else: # 树不完整
            tmp_trees.extend(news) # 加入新结果
            tmp_trees.sort(key=lambda x:x.cost-x.depth) # 排序
    # print("ans is ", min_cost)
    return optimal,min_cost

def next_level(tree,nonleaves,nonleaf_cost,leaf_cost):
    """生成下一层树"""
    news,is_end = [],False
    k,n = tree.max_depth,len(nonleaves)
    ak = nonleaves[k]
    assert ak,"展开数不能为0"
    # 三类节点
    sep,not_sep,whatever = [],[],[]
    for node in tree.last_layer:
        if node.position in nonleaf_cost:
            sep.append(node.position)
        elif node.position in leaf_cost:
            not_sep.append(node.position)
        else:
            whatever.append(node.position)
    # 产生新树
    if ak < len(sep): # 展开少，取 sep 子集，增加未取部分开销
        for choice in choose(sep,ak):
            new = tree.new_tree_by_positions(choice)
            new.cost += sum(nonleaf_cost[i] for i in sep if i not in choice)
            if new.cost <= min_cost:
                news.append(new)
    elif len(sep) <= ak <= len(sep)+len(whatever): # 展开适中，不增加开销
        choice = sep + whatever[:ak-len(sep)]
        news = [tree.new_tree_by_positions(choice)]
    else: # 展开多，取 not_sep 子集，增加选取部分开销
        not_sep.sort(key = lambda s: leaf_cost[s])
        choice = not_sep[:ak-len(sep)-len(whatever)]
        new = tree.new_tree_by_positions(sep+whatever+choice)
        new.cost += sum(leaf_cost[i] for i in choice)
        if new.cost <= min_cost:
            news.append(new)
    if nonleaves[k+1]==0:
        is_end = True
        if k+2<n: # 剩下节点合并
            for new in news:
                new.cost += sum(nonleaf_cost.get(node.position,0) for node in new.last_layer)
    return news,is_end

### 调试函数 ###
def get_operations(old,new):
    """获取变动信息：(拆分, 合并)"""
    f = lambda a,b,c:a.difference(b).intersection(c)
    old_leaves = set(node.position for node in old.leaves)
    new_leaves = set(node.position for node in new.leaves)
    old_nodes = set(node.position for node in old)
    new_nodes = set(node.position for node in new)
    sep = f(new_nodes,new_leaves,old_leaves)
    com = f(old_nodes,old_leaves,new_leaves)
    return sep,com

def is_nonleaves(nonleaves):
    """检查非叶节点序列是否有误"""
    if len(nonleaves)==0:return False
    if len(nonleaves)==1:return True
    return all([2*i>=j for i,j in zip(nonleaves[:-1],nonleaves[1:])])

def leaves_to_nonleaves(leaves):
    """叶节点序列 -> 非叶节点序列"""
    assert len(leaves)>0, "输入不能为空列表"
    # 非叶节点序列和总节点序列
    nonleaves,nodes = [1-leaves[0]],[1]
    for i in leaves[1:]:
        nodes.append(2*nonleaves[-1]) # 总节点数
        nonleaves.append(nodes[-1]-i) # 可用根节点数
    # 检查未项是否只剩叶节点
    assert nonleaves[-1]==0,"输入叶节点序列不完整"
    return nonleaves

def random_nonleaves_seq(n):
    """随机生成 n 层非叶节点序列"""
    assert n>0, "层数至少为1"
    if n==1: return [0]
    seq = [1]
    while len(seq)!=n-1:
        total = seq[-1] * 2
        seq.append(randint(1,total))
    return seq+[0]
nonleaves2leaves = lambda nonleaves:[0]+[2*a-b for a,b in zip(nonleaves[:-1],nonleaves[1:])]
random_leaves_seq = lambda n: nonleaves2leaves(random_nonleaves_seq(n))

def random_tree_positions(n,max_value=30)->list:
    """生成随机树，叶子取值范围为 [0,max_value]"""
    tree = BTree.random_binary_tree(n-1,max_value)
    return BTree.tree_to_positions(tree)