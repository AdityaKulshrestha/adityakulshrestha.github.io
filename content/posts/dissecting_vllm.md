+++
title = 'Dissecting vLLM'
date = 2026-04-16T19:26:27Z
draft = true
+++

Bojour!

If you are someone who is looking to explore more into vLLM and understand the codebase repository. This can be your introductory guide to get started with it. My goal in this blog is to take you through a prompt journey - how it gets processed when received by the vLLM server, taken to prefill and then battling for bandwidth among other requests along with managing its KV cache. 
Also we will be understanding how the fundamental concept of OS actually helped the developers to come up with such an efficient system.


## Defining the prompt.

Let's consider we have three prompts requests:
1. Write a story on a pompous king.
2. What is the root of 16?
3. What do you think about AI. 

For sake of simplicity, I will not consider the serving format. We will see how the engine actually takes the requests coming in and what process it follows.



## Step 1 - Instantiating the vLLM Engine
The step first is to instantiate the vLLM engine. The engine is basically the core component that takes care of loading the model, managing the KV cache and scheduling the requests.

The journey starts from [here](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py). As you can see in the code the model starts with defining the config (it includes what model name, any tensor parallelism, max batch size, max number of parallel requests to be processed) etc. This config and other details are used to create the `EngineArgs` which is a class to handle. 

During the bootup it calculates things such as how much KV Cache memory can be allocated, download the weights if not available, check for device, do torch compile or capture it in CUDA graph (in case of GPU) and do warmup across for different input and output shapes. For now think of it as getting the engine ready to deal with any kind of googly prompt coming its way to serve the best to you in terms of performance. You can find the core of the LLM Engine [here](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/async_llm.py)

The code components of LLM Engine are:
1. vLLM Config - Contains information about the model, cache, parallelism etc.
2. Input Processor - Turns raw inputs into vLLM Input class.
3. Engine Core Engine - Component that actually performs inference. It runs as a separate background process and the communication happens through `EngineCoreClient`. Think of Engine Core Engine as the cook, LLM Engine as the waiter and Engine Core Client as the communication channel like a phone call or digital order system.
4. Output Processor - Converts the output from the EngineCore into the format that can be sent back to the client.

Engine Core has several important components, we will covert few which are critical to understand:
1. Model Executor - Performs forward pass
2. Scheduler - Manages the queue (waiting and running), KV cache and policy (priority; generally the decode requests are given a higher priority). 



Once everything gets loaded up, we can see how it access the requests and make it go through the next set of process. 

## InputProcessor

The request after being received by the LLM Engine where is passed to `InputProcessor` which takes care of input request validation, tokenization and processing converting the input request to `EngineCoreReqeust` object.

The request is then sent off to the `EngineCoreClient` which takes the request and passes it to the `EngineCore`. The `EngineCore` takes the request and converts it to a `Request` object essentially passing it to the Scheduler


## Scheduler



## KV Cache Manager
The KV Cache Manager is responsible for managing the KV Cache memory. It works in similar way how the OS manages the memory allocation. Instead of writing out to a single page, it uses a concept of page table to manage the memory allocation. 

Current Problem - Normal memory allocation results in lots of fragmentation, non contiguous layout format and waste of memory.

Solution - Virtual Memory Paging (OS Concept). 

vLLM breaks the memory allocation (Key and Value) allocation into fixed-size blocks (aka pages). It devides the entire memory sequence into blocks capable enough to store 16 tokens.

To calculate the total number of blocks - vLLM engine calculate the size required to store the KV cache for 16 tokens for all the layers available in the model.

- 2 (for key and value) x block_size (16 tokens) x num_layers (decoder layers) x head_size (head dimension) x num_kv_heads (num of kv_heads) x data_type (float16/bfloat16)

For example: Llama 3 8B model
- 2 x 16 x 32 x 128 x 8 x 2 = 1,310,720 bytes = 8.388 MB

If we have the total memory for KV cache to be 31.42 GB
We can have 31.42 * 1024 / 2.09 = 15,464 blocks

15,464 * 16 = 247,424 tokens can be stored in the KV cache.



In the global memory the blocks can lie anyware, the mapping of the tokens to the block id is doing through `Block Table`

Example, consider a prompt of 48 tokens. By the definition that each Block can accomodate 16 tokens:
- Tokens 0-15: Block 1
- Tokens 16-31: Block 3
- Tokens 32-47: Block 7





KV Cache manager is responsible for the following things:
- Allocate new blocks
- Free blocks when sequence finish
- Maintain a free list
- Track which blocks are currently in use.

