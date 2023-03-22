#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import random
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


# In[ ]:


class TS_4_WHouseAlloc():
    def __init__(self, data, leaseCosts, Q, Y, n, nxbit, w=10000, K=10, lamda=0.5, max_epoch=1000):
        """data-仓库坐标 leaseCosts-租赁成本 Q-商家所选仓库个数 n-领域集合中的元素个数 nx-领域操作位数
        w-商家运输货物总量 K-运送货物单价 Y-初始样本数目 lamda-每周期仓库之间调货的平均次数"""
        self.N = data.shape[0]  # 云仓库的总数目
        self.Q = Q  # 商家所选仓库个数
        self.Y = Y  # 初始样本数目
        self.w = w  # 商家运输货物总量
        self.K = K  # 运送货物单价
        self.lamda = lamda # 每周期仓库之间调货的平均次数
        self.max_epoch = max_epoch # 最大迭代次数
        self.leaseCosts = leaseCosts # 每间仓库租赁成本
        self.disti  = self.calcuDisti(data)  # N个云仓库到商家的距离
        self.distij = squareform(pdist(data, metric='euclidean'))  # 计算N个仓库的距离矩阵 
        self.solutions = self.initSolutions() #初始化Y个解
        self.n = n  # 领域集合中的元素个数
        self.nxbit = nxbit  # 领域操作位数
        self.T = int(np.around(np.sqrt(n))) # 禁忌长度
        self.tabu = dict() # 初始化禁忌表
        self.nowV, self.nowS = self.initNowS_V() #初始化当前解,当前值
        self.bestV = np.inf #历史最优值
        self.bestS = None   #历史最优解
        return
    
    def calcuDisti(self, data): 
        """分别计算N个云仓库到商家的距离  data-仓库坐标"""
        disti = np.zeros(self.N)
        for i,p in enumerate(data):
            disti[i] = np.sqrt(np.square(p[0]) + np.square(p[1]))
        return disti
    
    def initSolutions(self):
        """初始化解"""
        solutions = np.zeros((self.Y, self.N), dtype=np.int8)
        for s in solutions:
            s[random.sample(range(self.N), self.Q)] = 1  # 随机选择仓库
        return solutions
    
    def initNowS_V(self):
        """初始化当前解 当前值"""
        nowV = np.inf
        nowS = None
        for s in self.solutions:
            v = self.efunc(s)
            if v < nowV:
                nowV = v
                nowS = s
        return nowV, nowS
    
    def efunc(self, solution):
        """目标函数 solution-某个解"""
        c = np.sum(self.disti[solution == 1]) + np.sum(self.leaseCosts[solution == 1]) # 总成本
        index = np.nonzero(solution)[0]
        temp  = 0
        for i in range(self.Q):
            for j in range(self.Q):
                temp += self.distij[index[i], index[j]]
        c += (self.lamda * self.K / self.Q) * temp
        return c
    
    def isInTabu(self, solution):
        """判断是否在禁忌表"""
        if tuple(solution) in self.tabu:
            return True
        else:
            return False
    
    def updateTabu(self):
        """每轮迭代结束前更新禁忌表"""
        delete = list()
        for k, v in self.tabu.items():
            v -= 1
            if v == 0:
                delete.append(k)
        if len(delete) > 0:
            for k in delete:
                del self.tabu[k]
        return 
    
    def localBestS_inTabu(self, x):
        """邻域中的局部最优解进入禁忌表"""
        self.tabu[tuple(x)] = self.T
        return
    
    def pick_from_Nx(self, x):  
        """从当前状态领域中挑选一个状态  x-当前解"""
        newx = x.copy()
        flag = True
        while flag:
            operateBits = random.sample(list(np.nonzero(x)[0]), self.nxbit)
            for i in operateBits:
                r = random.randint(1, self.N-1)
                newi = (i + r) % self.N
                while newx[newi] == 1:
                    newi = (newi+1) % self.N
                newx[newi] = 1
                newx[i] = 0                    
            flag = self.isInTabu(newx)
        return newx
    
    def getLacalBest_from_Nx(self, x):
        '''从领域集合中得到局部最优'''
        localBestV = np.inf  # 最优值
        localBestS = None    # 最优解
        for i in range(self.n):  # 从领域中挑选n个元素
            newx = self.pick_from_Nx(x)
            localV = self.efunc(newx)
            if localV < localBestV:
                localBestV = localV
                localBestS = newx
        return localBestV,localBestS
    
    def run(self):
        t = 0
        ts_iterate = list()
        while t < self.max_epoch:
            t += 1
            localBestV, localBestS = self.getLacalBest_from_Nx(self.nowS) #从邻域中获取局部最优值/解
            self.updateTabu() #更新禁忌表-禁忌长度减一
            self.localBestS_inTabu(localBestS) #局部最优解进入禁忌表
            self.nowS = localBestS  #更新当前解
            self.nowV = localBestV  #更新当前值
            if localBestV < self.bestV:  #判断是否更新历史最优
                self.bestV = localBestV
                self.bestS = localBestS
            ts_iterate.append(self.bestV)
        return self.bestV, self.bestS, ts_iterate


# In[ ]:




