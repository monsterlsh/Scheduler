#!/usr/bin/env python
# coding: utf-8

import numpy as np

# 计算每台机器上资源使用量
# 输入为t时刻各VM的cpu需求量以及VM放置情况
# 输出为t时刻各机器的资源使用总量
def ResourceUsage(cpu_t, x_t):
    #x_cput = np.empty(shape = [0, len(cpu_t)])  # 创建空矩阵
    #x_cput = np.zeros(shape=x_t.shape)
    
    # row = 0   #  利用矩阵索引取矩阵每一行元素，初值为0
    # for i in range(len(cpu_t)):    # 几行乘几次
    #     temp = x_t[row, :] * cpu_t[i]    #  对矩阵xt第0行所有元素乘以cpu[0]的值
    #     #x_cput = np.vstack((x_cput, temp))   #  按行合并矩阵，利用空矩阵实现第一次迭代
    #     #row = row + 1
    x_cput = x_t.copy().T
    x_cput = np.matmul(x_cput,cpu_t)
    CPU_t = np.sum(x_cput, axis = 0) # 各列之和，即各机器上的cpu总需求量
    
    return CPU_t




# 计算负载均衡开销算法
# 输入为t时刻各VM的cpu需求量以及VM放置情况
# 输出为t时刻的负载均衡开销
def CostOfLoadBalance(cpu_t, x_t):
    CPU_t = ResourceUsage(cpu_t, x_t)
    cost_bal = np.sum(np.square(CPU_t)) # 每个元素平方后求和
    
    return cost_bal




# 计算迁移开销算法
# 输入为t-1时刻和t时刻的VM放置情况以及VM单位迁移成本
# 输出为t时刻的迁移开销
def CostOfMigration(x_last, x_now, E):
    num_mig = np.count_nonzero(x_now - x_last) / 2 # 迁移次数，即数组相减后的非零元素个数/2
    cost_mig = num_mig * E
    
    return cost_mig




# 随机放置算法
# 输入为行数列数
# 输入为一个相同行列数的01矩阵，每行有且仅有一个1
def RandomPlacement(N, M):
    x_t = np.zeros((N, M), dtype = int) # 创建一个N行M列的全零矩阵
    for i in range(N): # 对每一行
        one = np.random.randint(0,M-1)
        x_t[i][one] = 1
    
    return x_t





# 调度算法
# 输入：采样数Z，随机放置重复次数上限K，时间窗口大小W，t0时刻VM部署情况x_t0，VM数量N，每个VM在W内每时刻的资源需求量cpu，VM单位迁移开销E，机器数量M，机器资源使用上限CPU_MAX，权衡参数a
# 输出：t1时刻的VM部署情况x_t1

# 当前为t0，需要获取下一时刻的放置方案，即t1
# 用二维01数组表示每时刻VM和机器之间的配对，每个VM的xkt是一个list，例如[0, 1, 0, 0, 0, 0]
# 若考虑多资源维度，则输入VM在窗口内其他资源的需求量，增加其他资源的负载均衡开销计算

def SchedulePolicy(N,M,W,cpu,x_t0,Z=100, K=10, E=3, CPU_MAX=300, a=0.6):
    # 初始化为不进行任何迁移时的开销，即只有负载均衡开销
    cost_min = 0
    for t in range(W):
        cpu_t = cpu[:, t] # 提取cpu矩阵中第t-1列，比如t=0即提取下一时刻各VM的cpu需求量
        cost_t = CostOfLoadBalance(cpu_t, x_t0) # 不进行任何迁移时在当前时刻的负载均衡开销
        cost_min = cost_min + cost_t # 所有时刻的负载均衡开销之和
    
    placement = x_t0 # 最终放置策略初始化为当前状况，即不进行任何迁移
    
    for z in range(Z):
        cost = 0 # 当前全套配置方案下的总成本
        x_last = x_t0 # 初始化，因为如果是第一次，则上一时刻则为x_t0，否则则为上一轮更新时所设定的x_last
        x_t1 = x_t0 # 初始化
        for t in range(W):
            # 选择满足约束的随机放置，若K次都不满足，则启用备用方案
            k = 0
            flag = 0
            while k < K and flag == 0:
                k = k+1
                x_t = RandomPlacement(N, M) # 在M个机器上随机放置N个VM
                cpu_t = cpu[:, t] # 提取当前时刻各VM的cpu需求量
                CPU_t = ResourceUsage(cpu_t, x_t) # 当前放置下各PM的资源使用量
                if (CPU_t < CPU_MAX).all(): # 都没有过载
                    flag = 1
                # 备用策略
                if k == K and flag == 0:
                    x_t = x_last
            
            if t == 0:
                x_t1 = x_t # 用于最后返回，MPC需要执行第一个时刻的放置
        
            # 计算此次迁移开销
            x_now = x_t
            cost_mig = CostOfMigration(x_last, x_now, E)
            
            x_last = x_t # 更新，用于下一时刻
            
            # 计算此次负载均衡开销
            cost_bal = CostOfLoadBalance(cpu_t, x_t)
            
            # 计算当前时刻放置的总开销
            cost_t = cost_bal + a * (M-1) * cost_mig
            # 计算当前放置方案下W窗口内总开销
            cost = cost + cost_t
        
        if cost < cost_min:
            cost_min = cost
            placement = x_t1
    
    # 执行完Z次随机采样后，按照最终placement矩阵的值进行调度
    return placement

