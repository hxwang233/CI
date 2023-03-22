#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import random
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


# In[ ]:


class MA_4_WHouseAlloc():
    def __init__(self, data, leaseCosts, Q, Y, matingRate, variationRate, n=200, nxbit=2, w=10000, K=10, lamda=0.5, max_epoch=1000):
        """data-仓库坐标 leaseCosts-租赁成本 Q-商家所选仓库个数 matingRate-交配概率 variationRate-变异概率
        w-商家运输货物总量 K-运送货物单价 Y-初始样本数目 lamda-每周期仓库之间调货的平均次数"""
        self.N = data.shape[0]  # 云仓库的总数目
        self.Q = Q  # 商家所选仓库个数
        self.Y = Y  # 初始种群数目
        self.w = w  # 商家运输货物总量
        self.K = K  # 运送货物单价
        self.lamda = lamda # 每周期仓库之间调货的平均次数
        self.max_epoch = max_epoch # 最大迭代次数
        self.leaseCosts = leaseCosts # 每间仓库租赁成本
        self.disti  = self.calcuDisti(data)  # N个云仓库到商家的距离
        self.distij = squareform(pdist(data, metric='euclidean'))  # 计算N个仓库的距离矩阵 
        self.solutions = self.initSolutions() #初始化Y个种群
        self.fitValues = np.zeros(Y)          #Y个种群的适应值
        self.calcuFitValues()
        self.matingRate    = matingRate    # 交配概率
        self.variationRate = variationRate # 变异概率
        self.nowV, self.nowS = self.initNowS_V() #初始化最佳适应值,最佳种群
        self.bestIndex     = [-float("inf"),-float("inf")]  # 当前最优解的索引
        self.n = n  # 禁忌局部搜索邻域集合中的元素个数
        self.nxbit = nxbit  # 禁忌局部搜索邻域操作位数
        self.T = int(np.around(np.sqrt(n))) # 禁忌长度
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
    
    def initNowS_V(self):
        """初始化最佳适应值 最佳种群"""
        nowV = np.inf
        nowS = None
        for s in self.solutions:
            v = self.efunc(s)
            if v < nowV:
                nowV = v
                nowS = s
        return nowV, nowS

    def calcuBest(self):  
        '''获取当前最优值和最优解'''
        bestGroup = self.solutions[np.argmin(self.fitValues)]
        bestFitValue = np.min(self.fitValues) 
        return bestGroup, bestFitValue
    
    def calcuFitValues(self):
        """计算适应值"""
        for sno,solution in enumerate(self.solutions):
            c = np.sum(self.disti[solution == 1]) + np.sum(self.leaseCosts[solution == 1]) # 总成本
            index = np.nonzero(solution)[0]
            temp  = 0
            for i in range(self.Q):
                for j in range(self.Q):
                    temp += self.distij[index[i], index[j]]
            c += (self.lamda * self.K / self.Q) * temp
            self.fitValues[sno] = c
        return
                   
    def select(self): # 最优化保存策略
        self.bestIndex[0] = np.argmax(self.fitValues)
        self.bestIndex[1] = np.argmin(self.fitValues)
        bestGroup = self.solutions[np.argmin(self.fitValues)]
        self.solutions[np.argmax(self.fitValues)] = bestGroup.copy()
        return self.solutions
    
    def mating(self): 
        '''交叉'''
        willmate = list()
        for k, group in enumerate(self.solutions):
            if k in self.bestIndex:
                continue # 当前最优解不参与交配
            r = random.random()
            if r < self.matingRate:
                willmate.append(group)
        if len(willmate) >= 2 : # 交配个体大于2才进行本轮交配
            if len(willmate) % 2 != 0:  # 交配个体为基数
                delIndex = random.randint(0,len(willmate)-1)  #随机剔除一个
                del willmate[delIndex]
            matingMap = random.sample(range(len(willmate)), len(willmate))
            for i in range(0, len(matingMap), 2):  # 交配过程
                x1 = matingMap[i]
                x2 = matingMap[i+1]
                new1 = random.sample(list(np.nonzero(willmate[x1]+willmate[x2])[0]), self.Q)               
                new2 = random.sample(list(np.nonzero(willmate[x1]+willmate[x2])[0]), self.Q)
                willmate[x1][np.nonzero(willmate[x1])[0]] = 0
                willmate[x1][new1] = 1
                willmate[x2][np.nonzero(willmate[x2])[0]] = 0
                willmate[x2][new2] = 1
        return
    
    def variation(self):
        '''变异'''
        for k, group in enumerate(self.solutions):
            if k in self.bestIndex:
                continue # 当前最优解不参与交配
            r = random.random()
            if r < self.variationRate:
                operateBits = random.sample(list(np.nonzero(group)[0]), 1)  #随机变异位
                for i in operateBits:
                    r = random.randint(1, self.N-1)  #随机变异位
                    newi = (i + r) % self.N
                    while group[newi] == 1:
                        newi = (newi+1) % self.N
                    group[newi] = 1
                    group[i] = 0
        return 

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

    def tabuLocalSearch(self, bestGroup, bestFitValue):
        '''利用禁忌算法对本轮最优解进行局部搜索'''
        self.tabu = dict() # 初始化禁忌表
        nowS = bestGroup
        for i in range(int(self.max_epoch/10)):
            localBestV, localBestS = self.getLacalBest_from_Nx(nowS) #从邻域中获取局部最优值/解
            self.updateTabu() #更新禁忌表-禁忌长度减一
            self.localBestS_inTabu(localBestS) #局部最优解进入禁忌表
            nowS = localBestS  #更新当前解
            nowV = localBestV  #更新当前值
            if nowV < bestFitValue:  #判断是否更新历史最优
                bestFitValue = nowV
                bestGroup = nowS
        return bestGroup, bestFitValue
    
    def run(self):  # 进化过程
        t = 0  # 当前迭代次数
        ma_iterate = list()
        while t < self.max_epoch:
            t += 1
            self.solutions = self.select()  #选择
            self.mating() #交配
            self.variation() #变异
            self.calcuFitValues() #计算适应值
            bestGroup, bestFitValue = self.calcuBest() #获取最优值和最优解
            bestGroup, bestFitValue = self.tabuLocalSearch(bestGroup, bestFitValue) #局部搜索
            if bestFitValue < self.nowV:  #判断是否更新历史最优
                i = np.argmax(self.fitValues)
                self.solutions[i] = bestGroup
                self.fitValues[i] = bestFitValue
                self.nowS = bestGroup
                self.nowV = bestFitValue
            ma_iterate.append(self.nowV)
        return self.nowV, self.nowS, ma_iterate


# In[ ]:




