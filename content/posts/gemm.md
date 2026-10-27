+++
title = 'Gemm'
date = 2026-06-13T14:34:17Z
draft = true
description = 'How CPUs and GPUs differ, and how matrix multiplication (GEMM) maps onto them.'
+++


## GPU vs CPU Architecture

Let's start with a question. What do you think? Between CPU and GPU which is more powerful? If you have guessed CPU, correct. If GPU, then you are not wrong entirely. GPUs as we call them general purpose unit are meant for following one instruction and applying it on huge amount of data something we call as Single Instruction Mulitple Data aka SIMD. It is like the pawns in the chess. We have too many of them but can move only one step forward at a time and can kill diagonally.

Central Processing Unit were built to perform too many operations simultaneously - run a backend server, manage transfering the data and executing database query. This involves juggling between multiple activited. Hence they were supposed to do Single Instruction Single Data (SISD). Though we have instructions set for CPUs which follows SIMD like AVX512 etc but on that sometime later. Think of CPUs like queen of chess - too powerful, can move anywhere and can perform multiple tasks. 

With this as the core difference let's move to the architecture of CPU vs GPU:

//TODO: Add GPU vs CPU architecture. 

## Logical vs Physical Hierarchy

A GPU architecture follows a hierarchy for its execution. Just like a army oranizational hierarchy where you have platoon, company, battallion and bridage (cors and commands as well). GPU architecture also follows a hierarchy. And just like each level of hierarchy is managed by a commandant in the army we have similar working principles for GPU. 

1. Threads: Lets start with the most basic unit (foot soldier) Threads. Threads if you know from your OS class are the smallest unit of execution. Each thread it known to perform one operation on the data. 
2. Warps: 


## Simple Matrix Multiplication

Let's say we have two matrices of shape [2,4] and [4,2] where M, N and K = 2,4 and 2 respectively. The output expected shape of the matrix multiplication is [2,2]. To do a basic matrix multiplication you have to follow three loops. 
- The outer two loops takes care of the elements from matrix A (i & j)
- The inner two loops takes care of the elements from matrix B (j & k)

```python
for i in range(M):
    for j in range(N):
        for k in range(K):
            C[i,j] += A[i,k] * B[k, j]
```

Here is a quick visualization of how it actually computes the matrix multiplication:

<iframe src="../../assets/gemm/matrix_multiplication_visualizer.html" width="100%" height="600px" style="border:none;"></iframe>


If we swap the rows:
<iframe src="../../assets/gemm/matmul_ikj_loop_visualizer.html" width="100%" height="600px" style="border:none;"></iframe>



## Slow Python
Now let's see why python is so slow:

- Dynamic Typing overhead: Python interpreter doesn't know before hand what datatype its working on. All the data is stored as PyObject which stores the metadata such as reference count, data and datatype. Since Python is dynamically typed, the interpreter must determine the object's type at runtime and dispatch the appropriate operation. Hence for every data, its supposed to load the data using pointer. Then check the datatype and then apply the operation. This is what we call as dyanmic overhead. 
- Looping: Python loops execute through the CPython bytecode interpreter. For each loop turn, the python interpreter run each instruction one by one - it loads the data, apply operation, update the output and repeat, adding significant overhead compared to compiled loops in C or NumPy..
- Memory Allocation: Memory allocation in Python is not straightforward. Python lists store pointers to Python objects rather than storing raw numeric values directly. Whenever the interpreter is supposed to fetch the data it performs following operation:
    ```
    Read pointer
    Follow pointer
    Read object
    Read type
    Read value
    ```
This makes it very slow as compared to numpy where the data stored is referenced by the beginning of the contiguous memory allocated to the array. Next it just stides/jumps by the datatype size to fetch the next value. 

<!-- Below is a bytecode for matrixmultiplication using the above naive loop. Observe  -->





## Resources
1. [Brrrviz](https://brrrviz.com/the-machine/memory-hierarchy)