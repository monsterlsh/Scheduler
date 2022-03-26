#!/usr/bin/env python
# coding: utf-8



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
    CPU_t = np.matmul(x_cput,cpu_t)
    # CPU_t = np.sum(x_cput, axis = 0) # 各列之和
    
    return CPU_t




def isAllUnderLoad(x_t, cpu_t, mem_t, CPU_MAX, MEM_MAX):
    CPU_t = ResourceUsage(cpu_t, x_t)
    MEM_t = ResourceUsage(mem_t, x_t)
    is_cpu = (CPU_t < CPU_MAX).all()
    is_mem = (MEM_t < MEM_MAX).all()
    is_all = is_cpu and is_mem # 所以资源都不过载才返回True
    
    return is_all




# 调度算法
# 输入：t0时刻VM部署情况x_t0，VM数量N，每个VM当前时刻的资源需求量cpu_t0和mem_t0，机器数量M，机器资源使用上限CPU_MAX和MEM_MAX
# 输出：t1时刻的VM部署情况x_t1

# 调度算法触发机制：每固定时间段检测是否有过载机器（即某机器的某资源量>75%），若有则触发
# CPU_MAX = MEM_MAX = 75
import numpy as np

def Sandpiper(x_t0, N, cpu_t0, mem_t0, M, CPU_MAX, MEM_MAX):
    # 计算当前各PM和VM的Vol值及VSR值
    CPU_t0 = ResourceUsage(cpu_t0, x_t0)
    MEM_t0 = ResourceUsage(mem_t0, x_t0)
    Vol_pm = 10000 / ((100 - CPU_t0) * (100 - MEM_t0)) # 注意每PM/VM的资源需求要<100%
    Vol_vm = 10000 / ((100 - cpu_t0) * (100 - mem_t0)) # 1*N矩阵
    VSR = Vol_vm/ mem_t0 # 1*N矩阵
    # print('mem_t0',mem_t0)
    # print('Vol_vm',Vol_vm)
    # print('VSR',VSR)
    # print('Vol_pm',Vol_pm)
    # 机器按Vol值按序排序，存储机器号
    pm_asc = Vol_pm.argsort()
    pm_desc = pm_asc[::-1]
    print('pm_desc',pm_desc)
    x_t = x_t0.copy() # 初始化
    # 按序对每台机器做迁出
    
    pm_desc.astype(int)
    for pm_out in pm_desc:
        # 将每台机器上的VM降序排序，存储VM号
        vm_in_pm = np.where(x_t0[:, pm_out] == 1)[0] # 该机器上VM的VM号
        VSR_in_pm = VSR[vm_in_pm] # 这些VM的VSR
        #print(vm_in_pm,VSR_in_pm)
        vm_VSR = np.array([vm_in_pm, VSR_in_pm]) # 二维数组，第一行为VM号，第二行为VM对应VSR值
        vm_asc = vm_VSR.T[np.lexsort(vm_VSR)].T # 按照VSR升序排序
        #vm_asc = vm_VSR[:,vm_VSR[1].argsort()]
        vm_desc = vm_asc[0, ::-1] # 获取降序排序后的VM号
        
        isMig = 0 # 该机器上是否已有VM迁移
        # print(vm_desc)
        # print(pm_asc)
        # print('CPU_t0',CPU_t0)
        # print('cpu_t0',cpu_t0)
        vm_desc.astype(int)
        pm_asc.astype(int)
        print('vm_desc',vm_desc)
        for vm in vm_desc: # 从VSR最大的VM开始被迁移
            if isMig:
                break
            for pm_in in pm_asc: # 从Vol最小的开始迁入
                if CPU_t0[int(pm_in)] + cpu_t0[int(vm)] <= CPU_MAX and MEM_t0[int(pm_in)] + mem_t0[int(vm)] <= MEM_MAX: # 循环结束条件是有机器放得下
                    x_t[int(vm)][int(pm_out)] = 0 # 迁出
                    x_t[int(vm)][int(pm_in)] = 1 # 迁入
                    isMig = 1
                    break
        print('ismig',isMig)
        # 循环结束条件是没有机器过载
        if isAllUnderLoad(x_t, cpu_t0, mem_t0, CPU_MAX, MEM_MAX):
            break
        
    return x_t


# ******************************以下只用于计算评价指标******************************


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




# 评价指标
# 执行当前决策后的负载均衡开销和迁移开销
# eval_bal = CostOfLoadBalance(cpu_t0, mem_t0, placement, b)
# eval_mig = CostOfMigration(x_t0, placement, mem_t0)

