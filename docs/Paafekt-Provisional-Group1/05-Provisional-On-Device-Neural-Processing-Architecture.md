# PROVISIONAL PATENT APPLICATION NO. 5 OF 6

## TITLE OF INVENTION

### ON-DEVICE NEURAL PROCESSING ARCHITECTURE FOR REAL-TIME SPATIAL COMPUTING APPLICATIONS ON RESOURCE-CONSTRAINED MOBILE HARDWARE

**INVENTOR:** Kishore Shivanna  
**APPLICANT:** Paafekt Inc.  
**FILING TYPE:** Provisional Patent Application  

---

## FIELD OF THE INVENTION

The present invention relates to computational architectures for mobile software applications, and more particularly to on-device software architectures that enable computationally intensive spatial computing tasks—including neural depth estimation, semantic segmentation, visual odometry, and real-time rendering—to execute efficiently on resource-constrained mobile computing devices lacking specialized depth-sensing hardware, without cloud offloading of computation.

---

## BACKGROUND OF THE INVENTION

The computational demands of three-dimensional spatial understanding applications—encompassing deep neural network inference for depth estimation and semantic segmentation, iterative geometric optimization for visual odometry, volumetric data fusion, real-time physically-based rendering, and interactive user interface operation—substantially exceed the capabilities of mid-range and older consumer mobile devices when approached naively. Prior art systems that achieve the required computational quality have done so by: (a) restricting to premium high-performance devices; (b) offloading computation to cloud servers; or (c) simplifying algorithms to the point where accuracy and visual quality are unacceptable.

The present invention addresses this gap by providing a software architecture that distributes computational workloads across the heterogeneous processing resources of a mobile device, manages computational budgets adaptively based on available hardware capability, schedules intensive computations to avoid blocking interactive user interface operation, and implements multiple computational efficiency techniques that collectively enable the full spatial computing pipeline to operate on mid-range hardware released as far back as several product generations.

---

## SUMMARY OF THE INVENTION

The present invention provides an on-device software architecture for spatial computing applications on mobile devices, comprising: a heterogeneous resource scheduler; an adaptive quality management system; an asynchronous processing pipeline with non-blocking user interface integration; a memory management system for large volumetric data structures; and a suite of computational efficiency techniques including model quantization, operator fusion, tiled processing, and progressive computation.

---

## DETAILED DESCRIPTION OF THE INVENTION

### 1. Heterogeneous Processing Resource Scheduler

FIG. 7 illustrates the heterogeneous processing resource scheduler.

Modern consumer mobile devices contain multiple distinct processing resources with different computational characteristics: the central processing unit (CPU), typically comprising multiple cores with different performance profiles; the graphics processing unit (GPU), optimized for massively parallel floating-point computation on regular data structures; and the neural processing unit (NPU) or dedicated hardware accelerator, optimized for low-precision matrix multiplication operations characteristic of neural network inference. Additionally, modern mobile operating systems provide mechanisms for specifying performance versus efficiency tradeoffs for CPU cores.

The heterogeneous resource scheduler assigns computational tasks to the processing resource most suited to their computational characteristics:

#### A. CPU Assignment

The CPU handles tasks requiring complex control flow, sequential data dependencies, and irregular memory access patterns, including: camera pose graph construction and management; bundle adjustment optimization (which is a sparse iterative numerical optimization with complex data dependencies); measurement computation; user interface logic; and application state management.

#### B. GPU Assignment

The GPU handles tasks requiring massively parallel computation on regular grid-structured data, including: depth map back-projection into 3D space; volumetric depth fusion; mesh generation from the volumetric representation; real-time physically-based rendering; shadow map computation; and screen-space rendering effects.

#### C. NPU Assignment

The NPU handles neural network inference tasks, including: feature extraction for visual odometry; neural depth estimation per frame; semantic image segmentation; object detection for architectural feature identification (door frames, windows); and real-time camera pose tracking feature matching.

The scheduler monitors resource utilization and thermal state of each processing unit in real-time, dynamically adjusting workload assignments to prevent thermal throttling and maintain sustained performance over extended scanning sessions.

### 2. Adaptive Quality Management

The adaptive quality management system monitors the measured computational capability of the specific device at runtime—not relying on device model categorization, but directly measuring achieved processing throughput and latency—and adapts computational parameters accordingly.

In one embodiment, the adaptive parameters include: (a) neural depth estimation model resolution (input image resolution to the depth estimation network, traded against inference latency); (b) neural segmentation model resolution; (c) volumetric grid resolution for depth fusion (voxel size, traded against memory usage and surface reconstruction detail); (d) physically-based rendering quality settings (number of shadow samples, ambient occlusion sample count, reflection approximation quality); and (e) frame selection rate during image acquisition.

Each parameter has a defined quality tier system with at least three tiers (e.g., high, medium, low). At application launch, the system runs a brief benchmark procedure measuring achieved throughput for each neural inference and rendering task. Based on benchmark results, the system assigns each adaptable parameter to the highest quality tier feasible within the available computational budget. During operation, the system monitors achieved frame rates and inference latencies and dynamically adjusts quality tiers to maintain target interactive frame rates.

### 3. Asynchronous Processing Pipeline

The computationally intensive spatial computing operations—particularly the full 3D reconstruction pipeline—are executed asynchronously on background processing threads, isolating them from the main user interface thread. The asynchronous architecture ensures that the user interface remains responsive and smooth at all times, even while intensive background computation is in progress.

The asynchronous pipeline is structured as a directed acyclic graph (DAG) of processing stages. Each stage consumes one or more inputs and produces one or more outputs. Stages execute concurrently whenever their input data is available and their assigned processing resource (CPU, GPU, NPU) is not occupied by a higher-priority stage. The pipeline scheduler manages stage execution priority, input/output data buffering between stages, and resource conflict resolution.

In one embodiment, the pipeline stages and their execution resources are:

- Frame capture and quality assessment → CPU
- Feature extraction → NPU
- Feature matching and relative pose estimation → CPU
- Neural depth estimation → NPU (highest priority during scanning)
- Depth fusion update → GPU (incremental update after each new frame)
- Bundle adjustment (global) → CPU (run as lower-priority background job)
- Semantic segmentation → NPU (lower priority, run when NPU is not needed for depth)
- User interface rendering → GPU (reserved highest-priority slice)

### 4. Memory Management for Volumetric Data

The three-dimensional volumetric data structures used in depth fusion (the voxel grid) can require substantial memory—potentially hundreds of megabytes for high-resolution room reconstructions. Consumer mobile devices have constrained RAM, and exceeding available RAM causes performance-degrading memory swapping or application termination.

The memory management system implements the following strategies:

#### A. Sparse Voxel Representation

The voxel grid stores data only for occupied voxels—voxels containing surface evidence. The vast majority of a room's interior volume is empty space. A sparse representation (using a hash map indexed by voxel coordinates) stores only the occupied voxels, reducing memory requirements by one to two orders of magnitude compared to a dense grid.

#### B. Tiled Processing

When the total memory requirement of the reconstruction task (even with sparse representation) exceeds available device RAM, the reconstruction is divided into spatial tiles that are processed sequentially, with each tile loaded into memory, processed, and stored back to device storage before the next tile is loaded.

#### C. Progressive Level of Detail

The reconstruction maintains multi-resolution representations of the reconstructed surfaces. During interactive visualization, low-resolution mesh representations are used for real-time rendering (lower memory requirement, higher rendering throughput), while high-resolution details are loaded progressively for the spatial region currently in view.

#### D. Memory Pressure Monitoring

The system monitors available system memory in real-time. When available memory falls below a threshold, the system selectively releases cached data structures (e.g., intermediate computation results) that can be recomputed if needed, in preference order (least critical cached data released first).

### 5. Model Quantization and Compression Pipeline

All neural network models used in the spatial computing pipeline are processed through a model preparation pipeline that produces optimized, on-device-executable model artifacts:

#### A. Post-Training Quantization

Model weights are quantized from 32-bit floating-point to 8-bit integer or 4-bit integer representation, reducing model storage size by 4x or 8x and substantially increasing inference throughput on NPU hardware that operates natively in integer arithmetic.

#### B. Structured Pruning

Neural network weight tensors are pruned to remove weights with low magnitude or low gradient contribution to model accuracy, introducing structured sparsity that can be exploited by hardware to skip computation.

#### C. Knowledge Distillation

Large, accurate teacher models generate training targets for smaller, efficient student models that are deployed on-device. The student models achieve accuracy close to the teacher with substantially fewer parameters and operations.

#### D. Operator Fusion

Adjacent neural network operations that can be mathematically combined without intermediate data materialization (e.g., convolution followed by batch normalization followed by activation) are fused into single hardware-level kernels, reducing memory bandwidth and scheduling overhead.

### 6. Progressive Computation and Interruptible Refinement

For computations that can benefit from progressive refinement—particularly global bundle adjustment and high-resolution depth fusion—the system implements an interruptible refinement architecture. The computation is structured to produce a valid (if approximate) result after a small number of iterations, and to progressively refine the result over additional iterations. At any point, the computation can be interrupted (e.g., to service a higher-priority rendering request) and resumed from the current state without loss of progress. This ensures that the system always has a valid reconstruction result available for visualization purposes, even before the full high-quality reconstruction is complete.

---

## CLAIMS

**Claim 1.** A mobile computing system for on-device spatial computing, comprising: a central processing unit; a graphics processing unit; a neural processing unit; non-transitory computer-readable storage; and a heterogeneous resource scheduler executing on the central processing unit and configured to: assign neural network inference tasks to the neural processing unit, assign parallel geometric computation tasks to the graphics processing unit, and assign sequential optimization and control flow tasks to the central processing unit; wherein the system executes a spatial computing pipeline including three-dimensional room reconstruction and furniture visualization entirely on the mobile computing system without cloud offloading.

**Claim 2.** The system of claim 1, further comprising an adaptive quality management module configured to measure device computational capability at runtime and adaptively adjust computational quality parameters based on measured capability.

**Claim 3.** The system of claim 2, wherein the adaptively adjusted computational quality parameters comprise at least one of: neural network input resolution, volumetric reconstruction grid resolution, and rendering quality settings.

**Claim 4.** The system of claim 1, wherein the spatial computing pipeline is structured as a directed acyclic graph of processing stages executing concurrently on different processing resources.

**Claim 5.** The system of claim 1, further comprising a sparse voxel representation for volumetric data that stores data only for occupied voxels, reducing memory requirements relative to dense volumetric representations.

**Claim 6.** The system of claim 1, further comprising a memory pressure monitoring module configured to selectively release cached data when available system memory falls below a threshold.

**Claim 7.** The system of claim 1, wherein neural network models in the spatial computing pipeline are quantized to reduced-precision integer arithmetic for efficient execution on the neural processing unit.

**Claim 8.** The system of claim 7, wherein the quantization reduces neural network model weight representation from 32-bit floating point to 8-bit or 4-bit integer representation.

**Claim 9.** The system of claim 1, wherein optimization computations in the spatial computing pipeline are structured as interruptible refinements that produce valid approximate results after a minimum number of iterations and refine results progressively upon additional computation.

**Claim 10.** The system of claim 1, wherein the spatial computing pipeline executes on mobile computing devices without LiDAR sensors.

**Claim 11.** A computer-implemented method for on-device spatial computing on a mobile device, comprising: assigning neural network inference tasks to a neural processing unit of the mobile device; assigning parallel geometric computation tasks to a graphics processing unit of the mobile device; assigning sequential optimization and control flow tasks to a central processing unit of the mobile device; executing assigned tasks concurrently on respective processing units; and producing a three-dimensional room model and furniture visualization output entirely on the mobile device without cloud computation.

**Claim 12.** The method of claim 11, further comprising benchmarking processing throughput at runtime and adaptively configuring computational quality parameters based on benchmark results.

**Claim 13.** The method of claim 11, further comprising monitoring thermal state of processing units and adjusting workload assignments to prevent thermal throttling.

**Claim 14.** The method of claim 11, further comprising tiling volumetric data processing into spatial segments when total memory requirements exceed available device RAM.

**Claim 15.** A non-transitory computer-readable medium storing instructions that, when executed, perform the method of claim 11.

---

## ABSTRACT

A mobile computing software architecture for on-device spatial computing that enables computationally intensive three-dimensional reconstruction, neural depth estimation, semantic segmentation, and real-time rendering to execute on resource-constrained consumer mobile devices without cloud offloading. A heterogeneous resource scheduler assigns computational tasks to the most suitable processing resource—CPU, GPU, or NPU—based on task computational characteristics. An adaptive quality management system measures device capability at runtime and configures computational parameters accordingly. An asynchronous pipeline structured as a directed acyclic graph maintains user interface responsiveness during intensive background computation. Memory management techniques including sparse voxel representation, tiled processing, and memory pressure monitoring handle large volumetric data within device memory constraints. Neural network models are optimized through quantization, pruning, and operator fusion for efficient mobile execution.
