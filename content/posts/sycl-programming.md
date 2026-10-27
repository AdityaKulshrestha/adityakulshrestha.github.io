---
title: 'SYCL Programming (Part 1): Knowing the hardware'
date: 2026-10-25T12:12:11Z
draft: false
description: 'Introduction to Intel GPU architecture (Xe) and SYCL Programming'
tags: [GPU, SYCL, Intel, Optimization, Kernel]
---


This blog is a part of a series of blogs on SYCL programming. This blog is an introduction to SYCL programming and the Intel Xe architecture. This part will cover details about the SYCL programming and how is it similar to CUDA, terminologies and analogies and how is it executed on Intel Xe architecture. 

This is not purely entry level blog for someone trying to get into SYCL programming. If you worked on CUDA or any other hardware accelerator or GPU, this can be a good starting point for you. For others, you can start by learning about GPU architecture in general. I suggest you to watch [this](https://www.youtube.com/watch?v=h9Z4oGN89MU) video for best visual understanding.


{{% sidenote %}}
A Domain-Specific Language (DSL) is a computer language tailored to a specific application domain or problem type, unlike a General-Purpose Language (GPL) like Python, Java, or C++ which is designed to build software across many domains. Some of the popular examples of DSL are Triton and CuTe DSL.
{{% /sidenote %}}


Do note that SYCL is hardware agnostic, you can use SYCL to program Intel, AMD or Nvidia (CPUs and GPUs). Thus making it a powerful vendor agnostic C++ domain specific language (DSL). For our scope we will be covering Intel Xe architecture. There are already plenty of articles and blog around Nvidia and AMD.



## Xe Architecture 
Intel GPU architecture has followed arch name prefix called Xe. There are different generation and variation of Xe architecture Xe-HP (High Performance - Data Center GPU), Xe Low Power Graphics (LPG for edge devices like Intel Lunar and Panther series) and the current Xe-HPG (High Performance graphics like Battlemage B70).


If you want to learn more on each architecture you can follow [this](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/intel-xe-gpu-architecture.html). For the scope of this blog and the upcoming we will cover Xe2 architecture. You don't need to know the family history of the Xe2 to work with it. However I encourage you to read about it to satisfy your epistemic curiosity.


### A. Physical Hierarchy
Now let's look at the physical hierarchy in the architecture and understand how the entire silicon is divided into units that work coherently on any kernel.

#### 1. Vector Engine
At the lowest fundamental level is the Xe Vector Engine or XVE. It is analogous to what a CUDA core is in Nvidia GPUs. The Xe2 architecture has 8 hardware threads per XVE. The XVE contains multiple Arithmetic Logical Units for different datatypes operations (FP64, FP32, FP16, BF16, INT32, INT16, INT8 etc). 

Each hardware thread within the XVE has a dedicated large-capacity high bandwidth SRAM called General Register File (GRF). Physically it's a single shared memory local to each XVE which is split up and distributed to the hardware threads by the dispatcher. The dispatcher is controlled at the Xe-Core level (we will cover this in next section). Though the assigning of threads and register is done at the Xe-Core level, the actual execution handling is done at the XVE level itself by the XVE Instruction Scheduler. 

{{% sidenote %}}
**What is Regular and Large GRF Mode?**

In Regular mode you can launch 8 threads per XVE with each threads having access to 128 register with 512 bits each. 
In Large mode you can launch 4 threads per XVE with each thread having access to 256 register with 512 bits. 

Regular mode and large mode depends on the type of kernel you are executing. It largely depends on the register pressure. 
{{% /sidenote %}}

For Battlemage B70 having Xe2 (Xe2-HPG) architecture. There are 
1. 128 registers with 512 bits each in Regular GRF Mode
2. 256 registers with 512 bits each in Large GRF Mode.


![Xe-Architecture](../../assets/xe/xve_xe2_battlemage.png "Xe2 Vector Engine (XVE) in Battlemage")


In Nvidia specifically Blackwell architecture, the GRF is equivalent to register file in a processing block. The XVE can be naively mapped to a processing block. XVE has its own private register files called GRF and Processing blocks have their own called Register File. The execution is handled by Warp Scheduler in Blackwell and in Xe it's handled by Instruction Scheduler. The physical allocations of registers and assignment of threads is done by SM for Nvidia and for Xe its done by Xe-Core's Thread Dispatcher.


#### 2. Xe-Core
A group of XVE engines make an Xe-Core. There are 8 XVE in one Xe-Core in Battlemage B70 graphics and a total of 32 Xe-Core. This gives us a total of 256 XVEs in total with a total hardware thread count of 2048. Xe-Core is analogous to Streaming Multiprocessor in Nvidia. 

Each Xe-Core comes with its own L1 cache (non-programmable) and shared memory (programmable). Similar to GPUs they share same physical block. L1 cache is managed by the hardware whereas the shared memory can be explicitly managed and programmed by the user. The L1 cache and shared memory is shared by 8 XVE engines. The total size for L1 in Battlemage is 256 KB, out of which 128 KB is programmable.

Along with the XVE Engine, Xe-Core also contain Intel Xe Matrix Extension (Intel XMX). Its counterpart to Nvidia's Tensor Core, specifically reserved for matrix multiplication and can be programmed by specialized APIs.
Together 8 XVE + 8 XMX comprises of 1 Xe-Core. 

![Xe-Core Architectural View](../../assets/xe/xe2_xe_core.png "Xe-Core architectural view")


#### 3. Render Slice
Each Render Slice contains the following core components:
- **4 Xe Cores** 
- **4 Ray Tracing Unit (RTUs)**
- **4 Texture Samplers**
- **1 Geometric Pipeline**
- **2 Pixel Backends**

We will not be covering the functionality of Texture Samplers, Geometric Pipelines and Pixel Backends since they are meant for 3D graphics rendering. 
In Xe2 architecture, each render slice contains 4 Xe-Core and each core having 8 XVE and XMX engine. This gives total of 32 XVE and XMX engine per render slice. 

![Xe2: Render Slice](../../assets/xe/render_slice.png "Xe2: Render Slice <small>[Source](https://www.pcmag.com/news/intels-battlemage-gpus-for-budget-desktops-conjure-up-competitive-value)</small>")

For scope of kernel programming we will not be looking at ray tracing and media engines.

#### 4. Xe Stack
A collection of Render Slice, global cache, memory controllers and **hardware contexts** is called Xe Stack. It contains a group of Render Slice sharing an L3 cache. 

> A hardware context is a physical unit which acts as a global feeder that reads the incoming commands from the CPU, decodes it and sends it off to Xe-Core dispatcher. It also handles the synchronization between the commands incase there is some data dependency between them.

Xe Stack can be mapped to one GPU die. But there is little difference, because Nvidia latest GPUs are massive, they have introduced two new abstraction called Texture Processing Cluster (TPC) and Graphics Processing Cluster (GPC). There is no direct mapping for them in Xe arch. 

There are 8 Render Slices in Xe2 architecture (Battlemage GPU) with each having 4 Xe-Core with each core having 8 XVE and 8 XMX engine. This totals to 256 XVE and 256 XMX engines. Each Xe Stack has its own L3 cache or Global Shared Memory. There is no L2 cache shared local to each render slice. Hence L2 and L3 represents the same global memory. Battlemage contains a 24MB of L3 cache. 


#### 5. Xe GPU
A collection of Xe Stack makes up one physical GPU. For most of the Xe architecture there is only one Xe Stack like how it is for Ada, Ampere and Hopper architecture in Nvidia. Except for Xe HPC which you can think of like the Blackwell architecture where two die are fused together through [2.5D packaging technology](https://nhanced-semi.com/2024/02/26/the-essentials-of-2-5d-advanced-packaging/).

Only Intel Data Center GPU Max series *(codename Ponte Vecchio ie PVC)* is known to have multi die configuration. It is part of Intel Xe-HPC architecture. Intel had shipped multi die GPU way ahead of Nvidia in 2023 itself with Ponte Vecchio (PVC), Nvidia introduced the multi-die chip in 2024 with its Blackwell GPUs.


![Intel Ponte Vecchio](../../assets/xe/intel_pvc.png "Intel Ponte Vecchio (PVC)<small>[Source](https://hc33.hotchips.org/assets/program/conference/day2/hc2021_pvc_final.pdf)</small>")



### B. Logical Hierarchy

Let's talk about how programmers actually code and how is it executed on Xe architecture. 

Intel relies on oneAPI stack for most of its silicon. For its GPU it uses SYCL which is similar to CUDA in Nvidia world. From a workload perspective it is used to program the instruction for the threads to execute on a set of data. 

Broadly the hierarchy has been broken down into these stages:

#### 1. Work-item
Work-item is the smallest fundamental unit of execution. It is like a logical thread (similar to Nvidia) which executes the instruction on a given data.

#### 2. Sub-Group
Work-items are grouped in sub-groups. These subgroups are similar to warp in Nvidia terminology. But unlike Nvidia where it is fixed i.e. 32 threads make one warp. In Intel you can configure it based on the supported config by the hardware. For Xe2 you can have subgroup in size of 16 or 32 work-items. 

{{% sidenote %}}
In simple terms, in SIMD you think operations in terms of vector (similar to how triton operates). You write instructions which are supposed to be executed on a vector. Whereas SIMT lets you work at a more granular level where you can instruct each thread to carry out its own instruction. In SIMT, each thread maintains its own Program Counter (PC) and call stack. In Nvidia, it's integrated within their hardware design.
{{% /sidenote %}}

Also one more to note here is, at the hardware level Xe2 architecture is fundamentally SIMD (Single Instruction Multiple Data). But SYCL allows programmer an abstraction of Single Instruction Multiple Threads (SIMT) through sub-groups. 

The subgroups are executed on Xe Core. 

#### 3. Work-Group
Work-Group is a collection of work-items (or you can consider sub-group as well). It is analogous to the concept of Blocks in Nvidia. The threads can be stacked in 1D, 2D or 3D format and can be accessed by respective indices along each dimension.

The global GPU hardware scheduler maps the SYCL work-groups to specific Xe-Cores. 


#### 4. nd-range
It is the highest abstraction. It is a collection of work-group which can be arranged in 1D, 2D or 3D format and can be accessed by respective indices along each dimension. It maps to the Grid terminology in Nvidia. 

![Hierarchy Visualization](../../assets/xe/sycl_execution_model.png "SYCL execution hierarchy")

---------------------------------------

### C. Summary

We have covered logical hierarchy of the work units and how are they stacked together in various abstraction to make the work easier for the user.  Below is the table that sums up the CUDA with SYCL notations.

{{% tablefig title="Table 1: Summary Logical Hierarchy" %}}
| CUDA | SYCL |
|---|---|
| thread | work-item |
| warp | sub-group |
| block | work-group |
| grid | nd-range |
{{% /tablefig %}}

Just like CUDA have different physical hierarchy that operates and process different logical abstractions. SYCL logical hierarchy is also executed by different physical hierarchy on Intel GPU side. The table below gives an overview of both worlds and maps them on similar level.

{{% tablefig title="Table 2: Summary Execution Hierarchy" %}}
| Logical Group                            | CUDA / NVIDIA Component                   | SYCL / Intel GPU Component    |
| ---------------------------------------- | ----------------------------------------- | ----------------------------- |
| Entire workload                          | Grid                                      | nd-range                      |
| Scheduling unit                          | Thread block                              | Work-group                    |
| Hardware processing unit                 | Streaming Multiprocessor (SM)             | Xe-core                       |
| Group executing together                 | Warp                                      | Sub-group                     |
| Hardware scheduling / execution resource | Warp scheduler and SM execution pipelines | Vector Engine hardware thread |
| Individual logical execution             | Thread                                    | Work-item                     |
| Parallel execution lanes                 | Warp lanes                                | SIMD lanes                    |
{{% /tablefig %}}


-----------------------------------------

*This blog is the part 1 of three blogs series covering Intel GPU architecture, programming in SYCL and beating oneDNN matrix multiplication kernel*


## References
1. [Intel Xe GPU Architecture](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/intel-xe-gpu-architecture.html) 
2. [Intel GPU Execution Hierarchy](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/thread-mapping-and-gpu-occupancy.html)
3. [oneAPI Essentials Training](https://github.com/oneapi-src/oneAPI-samples/tree/main/DirectProgramming/C%2B%2BSYCL/Jupyter/oneapi-essentials-training) 
4. [CUDA and SYCL Programming Comparison](https://oneapi-src.github.io/SYCLomatic/dev_guide/reference/compare-prog-models.html)