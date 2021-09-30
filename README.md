# animated-juliaset-generator
fast juliaset generation using numpy and numba jit
Numpy is the core library for scientific computing in Python. It provides a more efficient way for arrays and metrics computations spatially when combined with numba witch provides the feature of jit (just in time compiler), and it also allows to use CPU cores in parallel.  jit takes Python functions designated by particular annotations (the jit decorator ) and transforms as much as it can — via the LLVM (Low Level Virtual Machine) compiler — to efficient CPU and GPU (via CUDA for Nvidia GPUs and HSA for AMD GPUs) code. 
i was able to achieve about 1000 frames in 91 seconds (10.98 fps) on my intel core i7 7500U using the two cores .
