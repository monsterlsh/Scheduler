#!/usr/bin/env python
# coding: utf-8

import gc
import pyDOE
# 计算迁移开销算法
# 输入为t-1时刻和t时刻的VM放置情况以及VM的此刻mem需求
# 输出为t时刻的迁移开销
def CostOfMigration(x_last, x_now, mem_now):
    mig = ~(x_last == x_now).all(axis = 1) # 表示是否迁移的矩阵，即[True,False,False]表示VM_1迁移了
    #print('mig:',mig)
    cost_mig = sum(mig * mem_now) # 当前时刻每VM的单位迁移开销为其此刻mem占用
    
    return cost_mig


# 计算每台机器上资源使用量【优化版】
def ResourceUsageSimplify(cpu_t, x_t):
    x_cput = x_t.copy().T
    CPU_t = np.matmul(x_cput,cpu_t)
    #CPU_t = np.sum(x_cput, axis = 0) # 各列之和
    
    return CPU_t


# 计算负载均衡开销算法【新！计算每台机器上VM各自两两相乘之和】
def CostOfLoadBalanceSimplify(cpu_t, mem_t, x_t, b):
    cost_bal = 0
    for pm in range(x_t.shape[1]): # 对每一列，即每台机器
        vm = np.where(x_t[:, pm] == 1)[0] # 在该机器上的VM索引
        cpu = cpu_t[vm] # 对应VM的cpu
        mem = mem_t[vm]
        
        for i in range(len(vm)-1): # 每台VM与其他VM两两相乘
            c = cpu[i] * np.sum(cpu[i+1:])
            m = mem[i] * np.sum(mem[i+1:])
            cost_bal += c + b * m # cpu乘积+b*mem乘积，所有机器之和
    
    return cost_bal




def RandomGreedySimplify(M, a, b, u, v, x_last, cpu_t, mem_t):
    x_t = x_last.copy()
    CPU_t = ResourceUsageSimplify(cpu_t, x_last) # 当前放置下各PM的资源使用量
    MEM_t = ResourceUsageSimplify(mem_t, x_last)
    avg_CPU = np.sum(cpu_t) / M # 计算当前时刻最负载均衡情况下每台机器应承载的资源均量
    avg_MEM = np.sum(mem_t) / M
    
    
    
    cpumem = np.vstack((cpu_t, mem_t)).T # 合并为一个二维数组
    cpumem_desc = cpumem[np.lexsort(cpumem[:,::-1].T)] # 按照cpu的大小降序排序
    
    thresh_out = (avg_CPU ** 2 + b * avg_MEM ** 2) / 2 # 判断迁出机器候选集的标准
    thresh_in = thresh_out # 判断迁入机器候选集的标准，初始化
    cpu_sum = 0
    mem_sum = 0
    for i in cpumem_desc: # 对数组中的每一行，即每一个cpu-mem对
        cpu_sum = cpu_sum + i[0]
        mem_sum = mem_sum + i[1]
        if cpu_sum < avg_CPU and mem_sum < avg_MEM: # 还没达到均值
            temp = (i[0] ** 2 + b * i[1] ** 2) / 2
            thresh_in = thresh_in - temp
        else: # cpu或mem之和大于等于均值，则结束循环
            temp = ((avg_CPU - cpu_sum + i[0]) ** 2 + b * (avg_MEM - mem_sum + i[1]) ** 2) / 2
            thresh_in = thresh_in - temp
            break
    
    
    '''
    max_cpu = np.amax(cpu_t) # 找出当前container的最大资源需求
    max_mem = np.amax(mem_t)
    
    thresh_out = (avg_CPU ** 2 + b * avg_MEM ** 2) / 2 # 判断迁出机器候选集的标准
    thresh_in = ((avg_CPU ** 2 - avg_CPU * max_cpu) + b * (avg_MEM ** 2 - avg_MEM * max_mem)) / 2 # 判断迁入机器候选集的标准
    '''
    
    
    bal = CostOfLoadBalanceSimplify(cpu_t, mem_t, x_last, b)
    
    over = np.where(bal > thresh_out)[0] # 迁出候选集
    under = np.where(bal < thresh_in)[0] # 迁入候选集
    
    # 先不抽样了，全都进行迁出
    # 在over中随机选择u比例个机器作为迁出机器
    # source = np.random.choice(over, np.ceil(u * len(over)), replace = False) # 随机乱序选择，向上取整可保证至少选择1个
    print(f'\t\t over : {over} under{under}')
    # 对每台迁出机器随机选择v比例个VM迁出
    # for s in source:
    for s in over:
        mig_candi_s = np.where(x_last[:, s] == 1)[0] # 能被迁走的VM候选集
        # mig = np.random.choice(mig_candi, np.ceil(v * len(mig_candi_s)), replace = False) # 随机乱序选择
        n=1 
        samples=np.ceil(v*len(mig_candi_s))
        samples = int(samples)
        #print(samples)
        lhd = pyDOE.lhs(n, samples) # 拉丁超立方抽样，输出[0,1]
        mig_loc = lhd * len(mig_candi_s)
        mig_loc = mig_loc[:,0].astype(int) # 即要被迁移的contaienr的id在候选集中的位置
        mig = np.unique(mig_candi_s[mig_loc]) # 要被迁移的contaienr的id，去掉重复值
        print(f'\t\t\t候选集 {s}: 要被迁移的contaienr: {mig}')
        # 对每个迁移VM贪心选择最优迁入机器
        for m in mig:
            destination = s # 目标机器初始化为原本所在的机器
            
            for d in under: # 对每台低于均值的机器
                # 假设把m迁移到d上，带来的负载均衡开销降低值
                bal_d_cpu = cpu_t[m] * (CPU_t[s] - cpu_t[m] - CPU_t[d]) # 该VM资源量*（原机器上除该VM之外的资源总量-目标机器上原本的资源总量）
                bal_d_mem = mem_t[m] * (MEM_t[s] - mem_t[m] - MEM_t[d])
                bal_d = bal_d_cpu + b * bal_d_mem
                mig_m = a * (M-1) * mem_t[m] # 该VM的迁移开销，为此时mem
                
                max_bal = mig_m # 初始化为迁移开销，则保证每个负载均衡开销节省量都要大于迁移开销才能迁入
                if bal_d > max_bal: # 如果当前负载均衡节省量大于历史最大节省量，则迁入该机器
                    max_bal = bal_d
                    destination = d
            
            if destination != s: # 如果要迁
                x_t[m][s] = 0 # 把该VM从原来的机器上删除，添加到目标机器上
                x_t[m][destination] = 1
    
    return x_t




# 非完全随机放置算法【优化版】
def RandomGreedySimplify_old(M, a, b, u, v, x_last, cpu_t, mem_t):
    x_t = x_last.copy()
    CPU_t = ResourceUsageSimplify(cpu_t, x_last) # 当前放置下各PM的资源使用量
    MEM_t = ResourceUsageSimplify(mem_t, x_last)
    avg_CPU = np.sum(cpu_t) / M # 计算当前时刻最负载均衡情况下每台机器应承载的资源均量
    avg_MEM = np.sum(mem_t) / M
    
    any_over = np.where((CPU_t > avg_CPU) | (MEM_t > avg_MEM))[0] # 迁出候选集
    all_under = np.where((CPU_t < avg_CPU) & (MEM_t < avg_MEM))[0] # 迁入候选集
    #print(f'迁入 ={all_under}')
    # 在any_over中随机选择u比例个机器作为迁出机器
    source = np.random.choice(any_over, int(np.ceil(int(u * len(any_over)))), replace = False) # 随机乱序选择，向上取整可保证至少选择1个
    #print(f'source = {source}')
    # 对每台迁出机器随机选择v比例个VM迁出
    for s in source:
        mig_candi_s = np.where(x_last[:, s] == 1)[0] # 能被迁走的VM候选集
        mig = np.random.choice(mig_candi_s, int(np.ceil(int(v * len(mig_candi_s)))), replace = False) # 随机乱序选择
        #print(f'mig = {mig}')
        # 对每个迁移VM贪心选择最优迁入机器
        for m in mig:
            destination = s # 目标机器初始化为原本所在的机器
            
            for d in all_under: # 对每台低于均值的机器
                # 假设把m迁移到d上，带来的负载均衡开销降低值
                bal_d_cpu = cpu_t[m] * ( CPU_t[s] - cpu_t[m] - CPU_t[d]) # 该VM资源量*（原机器上除该VM之外的资源总量-目标机器上原本的资源总量）
                bal_d_mem = mem_t[m] * ( MEM_t[s] - mem_t[m] - MEM_t[d])
                bal_d = bal_d_cpu + b * bal_d_mem
                mig_m = a * (M-1) * mem_t[m] # 该VM的迁移开销，为此时mem
                #print(f'bal_d={ bal_d}, mig_m={mig_m} ')
                max_bal = mig_m # 初始化为迁移开销，则保证每个负载均衡开销节省量都要大于迁移开销才能迁入
                if bal_d > max_bal: # 如果当前负载均衡节省量大于历史最大节省量，则迁入该机器
                    #print(f'bal_d={ bal_d}, mig_m={mig_m} ')
                    max_bal = bal_d
                    destination = d
            #print(f'des ={destination} s = {s}')
            if destination != s: # 如果要迁
                x_t[m][s] = 0 # 把该VM从原来的机器上删除，添加到目标机器上
                x_t[m][destination] = 1
    
    return x_t


# 调度算法【优化版】
import numpy as np
from numpy import random

def SchedulePolicy(Z, K, W, u, v, x_t0, N, cpu, mem, M, CPU_MAX, MEM_MAX, a, b):
    # 本来应该先判断当前时刻是否负载不均衡需要迁移，但是考虑到cpu和mem资源都完全平均的可能性极小，不浪费时间了
    
    # 最小开销初始化为W内都不进行任何迁移时的开销，即只有负载均衡开销
    cost_min = 0
    for pm in range(x_t0.shape[1]): # 对每一列，即每台机器
        vm = np.where(x_t0[:, pm] == 1)[0] # 在该机器上的VM索引
        cpu_vm = cpu[vm] # 对应VM在W窗口的所有cpu
        mem_vm = mem[vm]
        
        cost_t = 0
        for i in range(len(vm)-1): # 每台VM与其他VM两两相乘
            c = np.sum(cpu_vm[i] * cpu_vm[i+1:])
            m = np.sum(mem_vm[i] * mem_vm[i+1:])
            cost_t += c + b * m
        cost_min += cost_t # 所有时刻的负载均衡开销之和
    
    # 最终放置策略初始化为当前状况，即不进行任何迁移
    placement = x_t0
    
    
    for z in range(Z):
        cost = 0 # 当前W内全套配置方案下的总成本
        x_last = x_t0 # 初始化，因为如果是第一次，则上一时刻则为x_t0，否则则为上一轮更新时所设定的x_last
        x_t1 = x_t0
        #print('z = %d ' % z)
        for t in range(W):
            # 选择满足约束的随机放置，若K次都不满足，则启用备用方案
            #print('t = %d' % t)
            k = 0
            flag = 0
            while k < K and flag == 0:
                #print('k = %d' % k)
                k = k+1
                cpu_t = cpu[:, t] # 提取当前时刻各VM的cpu需求量
                mem_t = mem[:, t]
                #print('before cpu_t=',cpu_t,'before mem_t=',mem_t)
                #print('before cpu mem using:',np.sum(cpu_t),np.sum(mem_t))
                x_t = RandomGreedySimplify(M, a, b, u, v, x_last, cpu_t, mem_t) # 放置算法
                #print('after random',(x_t==x_last).all())
                CPU_t = ResourceUsageSimplify(cpu_t, x_t)
                MEM_t = ResourceUsageSimplify(mem_t, x_t)
                #print(f'after CPU_t = {CPU_t.shape}  ')
                # max_cpu = np.amax(CPU_t)
                # max_mem = np.amax(MEM_t)
                #print('cpumax :',max_cpu,'mem max',max_mem)
                flags = ((CPU_t < CPU_MAX) & (MEM_t < MEM_MAX)).all()
                # test_cpu = np.where(CPU_t > CPU_MAX)[0]
                # test_mem = np.where(MEM_t > MEM_MAX)[0]
                # #test
                # test_last = ResourceUsageSimplify(cpu_t, x_last)
                #print(f'where cpu max machine id {test_cpu} ')
                #print(f'ove max  : cpu:{CPU_t [test_cpu ]} mem:{MEM_t[test_mem]} last cpu:{test_last[test_cpu]}')
                #print(test)
                if flags: # 都没有过载
                    #print('no overhead')
                    flag = 1
                # 备用策略
                if k == K and flag == 0:
                    # x_t = FallbackPlacement(cpu_t, x_last)
                   # print('k is there k= ',k)
                    x_t = x_last
            
            # 计算此次迁移开销
            cost_mig = CostOfMigration(x_last, x_t, mem_t)
            # 计算此次负载均衡开销
            cost_bal = CostOfLoadBalanceSimplify(cpu_t, mem_t, x_t, b)
            # 计算当前时刻放置的总开销
            cost_t = cost_bal + a * (M-1) * cost_mig
            # 计算当前放置方案下W窗口内总开销
            cost = cost + cost_t
            
            # 用于最后返回
            if t == 0:
                x_t1 = x_t # MPC需要执行第一个时刻的放置
            
            x_last = x_t # 更新，用于下一时刻
        
        if cost < cost_min: # 选择总开销低于完全不迁移的开销且最小的
            
            cost_min = cost
            placement = x_t1
            print('scheduler cost:',(placement ==x_t0).all() )
    # 执行完Z次随机采样后，按照最终placement矩阵的值进行调度
    del x_last
    del x_t1
    del x_t
    
    return placement,cost_min