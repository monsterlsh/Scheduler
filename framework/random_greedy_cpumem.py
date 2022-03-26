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
# 输入为t时刻各VM的cpu和mem需求量以及VM放置情况
# 输出为t时刻的负载均衡开销
def CostOfLoadBalance(cpu_t, mem_t, x_t, b):
    CPU_t = ResourceUsage(cpu_t, x_t)
    MEM_t = ResourceUsage(mem_t, x_t)
    cost_cpu = np.sum(np.square(CPU_t)) # 每个元素平方后求和（参照化简后公式）
    cost_mem = np.sum(np.square(MEM_t))
    cost_bal = cost_cpu + b * cost_mem
    
    return cost_bal




# 计算迁移开销算法
# 输入为t-1时刻和t时刻的VM放置情况以及VM的此刻mem需求
# 输出为t时刻的迁移开销
def CostOfMigration(x_last, x_now, mem_now):
    mig = ~(x_last == x_now).all(axis = 1) # 表示是否迁移的矩阵，即[True,False,False]表示VM_1迁移了
    cost_mig = sum(mig * mem_now) # 当前时刻每VM的单位迁移开销为其此刻mem占用
    
    return cost_mig





# 非完全随机放置算法
def RandomGreedy(M, a,b, u, v, x_last, cpu_t, mem_t):
    x_t = x_last
    CPU_t = ResourceUsage(cpu_t, x_last) # 当前放置下各PM的资源使用量
    MEM_t = ResourceUsage(mem_t, x_last)
    avg_CPU = np.sum(cpu_t) / M # 计算当前时刻最负载均衡情况下每台机器应承载的资源均量
    avg_MEM = np.sum(mem_t) / M
    
    # 将机器分为都大于+某项大于、都等于和都小于均值的3个集合
    # all_over = np.where((CPU_t > avg_CPU) & (MEM_t > avg_MEM)) # 存储的是满足条件的元素的索引，即机器编号
    any_over = np.where((CPU_t > avg_CPU) | (MEM_t > avg_MEM))[0]
    # one_over = np.setdiff1d(any_over, all_over) # 在any_over中但不在all_over中的元素，即只有一项大于的
    all_equal = np.where((CPU_t == avg_CPU) & (MEM_t == avg_MEM))[0]
    all_under = np.where((CPU_t < avg_CPU) & (MEM_t < avg_MEM))[0]
    
    # 在any_over中随机选择u比例个机器作为迁出机器
    num_source = max(int(u * len(any_over)), 1) # 至少选一个
    #print('test:',any_over,num_source)
    source = np.random.choice(any_over, num_source, replace = False) # 随机乱序选择num_source个元素
    
    # 对每台迁出机器随机选择v比例个VM迁出
    mig = np.empty(shape = 0, dtype = int) # 所有要迁移的VM
    for s in source:
        s_x = x_last[:, s] # 该机器
        num_s = np.sum(s_x) # 该机器上VM总数
        num_mig_s = max(int(v * num_s), 1) # 迁移VM个数，至少迁一个
        mig_candi_s = np.where(s_x == 1)[0] # 能被迁走的VM候选集
        mig_s = random.choice(mig_candi_s, num_mig_s, replace = False) # 随机乱序选择num_mig个元素
        mig = np.append(mig, mig_s)
    
    # 对每个迁移VM贪心选择最优迁入机器（随机顺序）
    for m in mig:
        min_bal = CostOfLoadBalance(cpu_t, mem_t, x_t, b) # 初始化为不迁移该VM的负载均衡指标
        last_d = np.argmax(x_last[m])
        destination = last_d # 初始化为原本所在的机器
        
        for d in all_under: # 对每台低于均值的机器
            # 假设把m迁移到d上
            x_test = x_t
            x_test[m][last_d] = 0
            x_test[m][d] = 1
            bal_d = CostOfLoadBalance(cpu_t, mem_t, x_test, b) # 进行该迁移后均衡指标
            
            mig_m = a * (M-1) * mem_t[m]
            if (bal_d + mig_m) < min_bal: # 如果迁移后总值低于不迁移，单位迁移开销为该VM此时的mem
                min_bal = bal_d
                destination = d
        
        if destination != last_d: # 如果要迁
            x_t[m][last_d] = 0 # 把该VM从原来的机器上删除，添加到目标机器上
            x_t[m][destination] = 1
    
    return x_t




# 调度算法
# 输入：采样数Z，随机放置重复次数上限K，时间窗口大小W，迁出机器占总超过均值机器的比例u，迁出VM占每台迁出机器的比例v，t0时刻VM部署情况x_t0，VM数量N，每个VM在W内每时刻的资源需求量cpu和mem，机器数量M，机器资源使用上限CPU_MAX和MEM_MAX，权衡参数a和b
# 输出：t1时刻的VM部署情况x_t1

# 改动：增加输入mem需求量、cpu与mem的平衡因子和mem使用上限，增加对mem负载均衡开销计算，修改单位迁移开销常数E为每时刻每VM的变量

import numpy as np
from numpy import random

def SchedulePolicy(Z, K, W, u, v, x_t0, N, cpu, mem, M, CPU_MAX, MEM_MAX, a, b):
    # 初始化为不进行任何迁移时的开销，即只有负载均衡开销
    cost_min = 0
    for t in range(W):
        cpu_t = cpu[:, t] # 提取cpu矩阵中第t-1列，比如t=0即提取下一时刻各VM的cpu需求量
        mem_t = mem[:, t]
        cost_t = CostOfLoadBalance(cpu_t, mem_t, x_t0, b) # 不进行任何迁移时在当前时刻的负载均衡开销
        cost_min = cost_min + cost_t # 所有时刻的负载均衡开销之和
    
    placement = x_t0 # 最终放置策略初始化为当前状况，即不进行任何迁移
    
    for z in range(Z):
        cost = 0 # 当前全套配置方案下的总成本
        x_last = x_t0 # 初始化，因为如果是第一次，则上一时刻则为x_t0，否则则为上一轮更新时所设定的x_last
        x_t1 = x_t0
        
        for t in range(W):
            print(f' in random_greddy z = {z} t = {t}')
            # 选择满足约束的随机放置，若K次都不满足，则启用备用方案
            k = 0
            flag = 0
            while k < K and flag == 0:
                k = k+1
                cpu_t = cpu[:, t] # 提取当前时刻各VM的cpu需求量
                mem_t = mem[:, t]
                x_t = RandomGreedy(M, a,b, u, v, x_last, cpu_t, mem_t) # 放置算法
                CPU_t = ResourceUsage(cpu_t, x_t)
                MEM_t = ResourceUsage(mem_t, x_t)
                
                if ((CPU_t < CPU_MAX) & (MEM_t < MEM_MAX)).all(): # 都没有过载
                    flag = 1
                # 备用策略
                if k == K and flag == 0:
                    # x_t = FallbackPlacement(cpu_t, x_last)
                    x_t = x_last
        
            # 计算此次迁移开销
            x_now = x_t
            cost_mig = CostOfMigration(x_last, x_now, mem_t)
            # 计算此次负载均衡开销
            cost_bal = CostOfLoadBalance(cpu_t, mem_t, x_t, b)
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
    return placement,cost_min




# 评价指标
# 执行当前决策后的负载均衡开销和迁移开销
# eval_bal = CostOfLoadBalance(cpu_t0, mem_t0, placement, b)
# eval_mig = CostOfMigration(x_t0, placement, mem_t0)

