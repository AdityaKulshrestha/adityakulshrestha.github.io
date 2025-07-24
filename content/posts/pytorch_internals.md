+++
title = 'Pytorch_internals'
date = 2025-07-22T14:12:54Z
draft = false
+++

# 

Ever wondered what goes inside when you actually call torch.compile(model)?
There are a bunch of individual components that work together in a sequential manner to squeeze out the best performance from the GPU. Each component passes on some intermediate packets called *Intermediate Representation (IRs)* (more about this later) to the next component in the sequence. 
Let's take a look what are the these components at an overview level and then we will dive deep into each one of them.

**torch.compile components**
- TorchDynamo: The fronend responsible for intercepting the Python code and convert it to graphs.
- AOTAutograd
- PrimTorch - It decomposes complex PyTorch operations into simpler, frequently used "primitive" operations.
- TorchInductor 
- Triton(GPU), C++(CPU), CUDA (GPU)


![Components Overview](./assets/pytorch_internals/components_overview.avif)


### 1. TorchDynamo
TorchDynamo is the first and foremost component in the sequence which is responsible for intercepting the Python code. 
It intercepts the Python bytecode at runtime and rewrites blocks of user code into graphs. This involves extracting subgraphs containing PyTorch operations while leaving the non-PyTorcch code untouched. 

*Shape Polymorphism* - 

*Fallback* - If the code can't be converted into graphs, it safely fall backs to PyTorch's eager execution.

*Intermediate Representation (IR)* - 


### 2. Ahead-Of-Time Autograd (AOTAutograd)
As the name suggests AOTAutograd is responsible for differentiation and backpropagation graph generation.
it generates the backware computation graph (needed for the gradients and training) from the captured forward graph (passed by the TorchDynamo). However it only captures the backward graph and doesn't apply the graph level optimization.


### 3. PrimTorch
It breaks down the complex PyTorch operations into simpler (primitive operations). This breaking down actually helps optimizing the graph further. It accepts a FX Graph and decomposes the operations into simpler operations.


### 4. TorchInductor
It further optimizes the graph and generates code to finally run on the hardware. It takes a simplified computation graphs and generates hihgly optimized low level code for the target hardware (CPU, HPU, GPU).

It also determines hardware level optimizations such as memory planning, tiling etc.

Due to multiple hardware adoption, torchinductor supports multiple backend
based on the target hardware. 

| Backend | Description | 
|---------|-------------|
| Inductor| Default backend: highly optimized for CPUs and GPUs |
| Eager   | Runs the model without the graph capture, no optimization happens in this mode |
| aot_eager| It applies the AutoAutograd to capture the graph but doesn't apply any further backend optimization |
| cudagraphs | Leverages CUDA Graphs for reduces CPU overhead |
| ipex  | Uses Intel Extension for PyTorch for CPU-optimized execution |
| onnxrt | Uses ONNX runtime for acceleration on CPU/GPU |
| torch_tensorrt | TensorRT-backend for high-speed inference on Nvidia-GPUs | 
| tvm | Uses Apache TVM compiler for cross hardware inference | 
| openvino | Uses Intel OpenVINO for accelerated inference on supported Intel hardware | 


Now if you see here the output code for the torchinductor is nothing but Python. Seems counterintuitive right? We ingest in Python code and burps out Python code but the output Python code is faster. 



NOTE (DO THIS BEFORE PUBLISHING) - 
- Add a section about how each component was used in eager mode.
- Explain more about CPU overhead in cudagraphs. 

## Debugging

Now that we have understood what are the components in torch.compile. How can we understand how each component is doing and which component is failing when we do torch.compile.

*The best environment variable for debugging is*
`TORCH_COMPILE_DEBUG=1`

### TorchDynamo logging





#### References 
1. [PyTorch 2.0 Live Q&A Series: PT2 Profiling and Debugging](https://www.youtube.com/live/1FSBurHpH_Q?si=NMWkZYZx1FaNQxYs)
2. [](https://blog.ezyang.com/2024/11/ways-to-use-torch-compile/)
3. [torch.compile, the missing manual](https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit?tab=t.0)
4. [PyTorch Logging](https://docs.pytorch.org/docs/stable/logging.html)
5. [Debugging PyTorch Memory with Snapshot](https://zdevito.github.io/2022/08/16/memory-snapshots.html)
6. [pytorch.compile docs](https://docs.pytorch.org/docs/stable/torch.compiler.html)