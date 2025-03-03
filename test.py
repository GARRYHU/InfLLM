import torch
import time

# 确保使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设定数据的大小
data_size = 200 * 1024 * 1024  # 100MB的数据
# 在主存中创建数据（CPU Tensor）
data_cpu = torch.rand(data_size).to("cpu")

# 预先分配显存
data_gpu = torch.empty(data_size, device=device)

# 重复读取的次数
num_iterations = 20

# 测试从主存读取数据到 GPU 显存的同步时间
for _ in range(num_iterations):
    start_time = time.perf_counter()
    # 从主存中读取到 GPU 显存，采用同步方式
    data_gpu.copy_(data_cpu)  # 使用 copy_ 确保是同步执行
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(duration*1000)

# 计算总时间和平均时间
total_time = end_time - start_time
average_time = total_time / num_iterations

print(f"总耗时: {total_time:.6f}秒")
print(f"平均每次读取时间: {average_time * 1e6:.2f}微秒")
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/2/28 03:00
# @Author  : Garry
# @File    : test.py
# @Software: PyCharm
# @Project : InfLLM
