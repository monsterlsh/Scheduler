#!/usr/bin/env python
# coding: utf-8
import numpy as np


# 计算每台机器上资源使用量
# 输入为t时刻各VM的cpu需求量以及VM放置情况
# 输出为t时刻各机器的资源使用总量
def ResourceUsage(cpu_t, x_t):
    # x_cput = np.empty(shape = [0, len(cpu_t)])  # 创建空矩阵
    # row = 0   #  利用矩阵索引取矩阵每一行元素，初值为0
    # for i in range(len(cpu_t)):    # 几行乘几次
    #     temp = x_t[row, :] * cpu_t[row]    #  对矩阵xt第0行所有元素乘以cpu[0]的值
    #     x_cput = np.vstack((x_cput, temp))   #  按行合并矩阵，利用空矩阵实现第一次迭代
    #     row = row + 1
    
    # CPU_t = np.sum(x_cput, axis = 0) # 各列之和，即各机器上的cpu总需求量
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





# 非完全随机放置算法
def RandomGreedy(M, E, a, u, v, x_last, cpu_t):
    x_t = x_last
    CPU_t = ResourceUsage(cpu_t, x_last) # 当前放置下各PM的资源使用量
    avg_CPU = np.sum(cpu_t) / M # 计算当前时刻最负载均衡情况下每台机器应承载的资源均量
    
    # 将机器分为大于等于和小于均值的三个集合
    over = np.where(CPU_t > avg_CPU) # 存储的是满足条件的元素的索引，即机器编号
    equal = np.where(CPU_t == avg_CPU)
    under = np.where(CPU_t < avg_CPU)
    
    # 在over中随机选择u比例个机器作为迁出机器
    num_source = max(int(u * len(over)), 1) # 至少选一个
    source = random.choice(over, num_source, replace = False) # 随机乱序选择num_source个元素
    
    # 对每台迁出机器随机选择v比例个VM迁出
    mig = np.empty(shape = 0, dtype = int) # 所有要迁移的VM
    for s in source:
        s_x = x_last[:, s] # 该机器
        num_s = np.sum(s_x) # 该机器上VM总数
        num_mig_s = max(int(v * num_s), 1) # 迁移VM个数，至少迁一个
        mig_candi_s = np.where(s_x == 1) # 能被迁走的VM候选集
        mig_s = random.choice( num_mig_s, replace = False) # 随机乱序选择num_mig个元素
        mig = np.append(mig, mig_s)
    
    # 对每个迁移VM贪心选择最优迁入机器（随机顺序）
    for m in mig:
        min_bal = CostOfLoadBalance(cpu_t, x_t) # 初始化为不迁移该VM的负载均衡指标
        last_d = np.argmax(x_last[m])
        destination = last_d # 初始化为原本所在的机器
        
        for d in under: # 对每台低于均值的机器
            # 假设把m迁移到d上
            x_test = x_t
            x_test[m][last_d] = 0
            x_test[m][d] = 1
            bal_d = CostOfLoadBalance(cpu_t, x_test) # 进行该迁移后均衡指标
            
            mig_m = a * (M-1) * E
            if (bal_d + mig_m) < min_bal: # 如果迁移后总值低于不迁移
                min_bal = bal_d
                destination = d
        
        if destination != last_d: # 如果要迁
            x_t[m][last_d] = 0 # 把该VM从原来的机器上删除，添加到目标机器上
            x_t[m][destination] = 1
    
    return x_t





import numpy as np
from numpy import random

def SchedulePolicy(Z, K, W, u, v, x_t0, N, cpu, E, M, CPU_MAX, a):
    '''
    调度算法采样数Z
    随机放置重复次数上限K
    时间窗口大小W
    迁出机器占总超过均值机器的比例u
    迁出VM占每台迁出机器的比例v
    t0时刻VM部署情况x_t0
    VM数量N
    每个VM在W内每时刻的资源需求量cpu
    VM单位迁移开销E
    机器数量M
    机器资源使用上限CPU_MAX
    权衡参数a
    return t1时刻的VM部署情况x_t1

    当前为t0，需要获取下一时刻的放置方案，即t1
    用二维01数组表示每时刻VM和机器之间的配对，每个VM的xkt是一个list，例如[0, 1, 0, 0, 0, 0]
    若考虑多资源维度，则输入VM在窗口内其他资源的需求量，增加其他资源的负载均衡开销计算
    '''

    # 初始化为不进行任何迁移时的开销，即只有负载均衡开销
    cost_min = 0
    for i in range(W):
        # cpu_t = np.array([cpu1, cpu2, cpu3, ..., cpuN]) # W内每时刻t时各VM的资源需求量
        cpu_t = cpu[:, t] # 提取cpu矩阵中第t-1列，比如t=0即提取下一时刻各VM的cpu需求量
        cost_t = CostOfLoadBalance(cpu_t, x_t0) # 不进行任何迁移时在当前时刻的负载均衡开销
        cost_min = cost_min + cost_t # 所有时刻的负载均衡开销之和
    
    placement = x_t0 # 最终放置策略初始化为当前状况，即不进行任何迁移
    
    for z in range(Z):
        cost = 0 # 当前全套配置方案下的总成本
        x_last = x_t0 # 初始化，因为如果是第一次，则上一时刻则为x_t0，否则则为上一轮更新时所设定的x_last
        x_t1 = x_t0
        
        for t in range(W):
            # 选择满足约束的随机放置，若K次都不满足，则启用备用方案
            k = 0
            flag = 0
            while k < K and flag == 0:
                k = k+1
                cpu_t = cpu[:, t] # 提取当前时刻各VM的cpu需求量
                #TODO 处理修改Greedy
                x_t = RandomGreedy(M, E, a, u, v, x_last, cpu_t) # 放置算法
                CPU_t = ResourceUsage(cpu_t, x_t)
                
                if (CPU_t < CPU_MAX).all(): # 都没有过载
                    flag = 1
                # 备用策略
                if k == K and flag == 0:
                    # x_t = FallbackPlacement(cpu_t, x_last)
                    x_t = x_last
        
            # 计算此次迁移开销
            x_now = x_t
            cost_mig = CostOfMigration(x_last, x_now, E)
            # 计算此次负载均衡开销
            cost_bal = CostOfLoadBalance(cpu_t, x_t)
            # 计算当前时刻放置的总开销
            cost_t = cost_bal + a * (M-1) * cost_mig
            # 计算当前放置方案下W窗口内总开销
            cost = cost + cost_t
            
            # 用于最后返回
            if t == 0:
                x_t1 = x_t # MPC需要执行第一个时刻的放置
            
            x_last = x_t # 更新，用于下一时刻
        
        if cost < cost_min:
            cost_min = cost
            placement = x_t1
    
    # 执行完Z次随机采样后，按照最终placement矩阵的值进行调度
    return placement




# # 评价指标
# # 执行当前决策后的负载均衡开销和迁移开销
# # 输入：下一时刻的各VM的资源需求量cpu_t1，每次调度得出的放置决策placement，迁移之前的放置情况x_t0，单位迁移开销E
# eval_bal = CostOfLoadBalance(cpu_t1, placement)
# eval_mig = CostOfMigration(x_t0, placement, E)

