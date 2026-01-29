# SHA-512 OpenCL Implementation

This repository provides an optimized implementation of the SHA-512 hash function in OpenCL, designed for high-throughput computation on GPUs and other parallel devices. The code is fully unrolled for performance, leveraging OpenCL intrinsics to minimize overhead and maximize execution speed. It focuses on pure compression without handling padding, making it ideal for scenarios like HMAC computations where padding is pre-applied or for single-block compressions.

## Features

- **High Throughput Design**: Fully unrolled 80 rounds of the SHA-512 algorithm, using low-level instructions for rapid execution.
- **Optimized for OpenCL**: Utilizes intrinsics such as `rotate()` for bit rotations and `bitselect()` for branchless Ch/Maj functions, reducing computational overhead.
- **Efficient Message Scheduling**: Computes the message schedule on-the-fly with a rolling window (W16 to W32), avoiding the need for a full 80-word array and minimizing private memory/register usage.
- **Pure Compression Focus**: Expects input as pre-formatted 16x `ulong` words per block. The `sha512_hash_two_blocks_message()` function compresses exactly two blocks (256 bytes) without padding, suitable for HMAC prefixes with completed padding or single compressions.
- **Low Memory Consumption**: Reuses variables in the schedule to keep memory footprint small, enabling better performance on resource-constrained devices.
- **Performance Claim**: This is one of the fastest SHA-512 implementations available for OpenCL, optimized for parallel workloads like cryptographic mining, password cracking, or bulk hashing.

## Usage

### Prerequisites
- OpenCL-compatible hardware (e.g., GPU from NVIDIA, AMD, or Intel).
- OpenCL development environment (e.g., via PyOpenCL for Python integration or direct C/C++ host code).
- The input message **must be pre-padded and formatted as big-endian 64-bit words** if necessary (the code assumes host-prepared blocks) (important).

### Integration Example
Save the provided code as `sha512.cl`. In your host application (e.g., Python with PyOpenCL):

```python
import pyopencl as cl

# Load the kernel code from file
with open('sha512.cl', 'r') as f:
    kernel_code = f.read()

# Set up OpenCL context and queue
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
program = cl.Program(ctx, kernel_code).build()

# Example: Add a kernel to use sha512_hash_two_blocks_message
# (Adapt as needed for your use case)
```

### Benchmark Results


**OpenCL Device:** NVIDIA RTX 4090

| Block | N | Time (ms) | Compress Ops/s | MH/s |
|--------|---|-----------|----------------|-------|
| 1-bloco | 100,000,000 | 21384.023 | 9,577,243,790 | 9577.2 |

With these optimizations, we achieve impressive benchmarks of up to 10 billion compressions per second (in tests that exclude kernel launch overhead and rely solely on the global ID to compute the exact function result, avoiding any timing contamination from input/output buffer I/O).

**Benchmark Code**
```python
# bench_sha512_manual_loops.py
import os, time
import numpy as np
import pyopencl as cl

def load_text(*paths):
    for p in paths:
        if p and os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
    return ""

common_code = load_text("kernel/common.cl", "common.cl")
sha512_code = load_text("kernel/sha512.cl", "sha512.cl")
if not sha512_code:
    raise RuntimeError("Não achei sha512.cl (tentei kernel/sha512.cl e sha512.cl).")

kernel_code = r"""
__kernel void bench_comp1(__global ulong *out) {
    uint gid = get_global_id(0);
    __private ulong H[8];
    __private ulong M[16]={((ulong)gid)}; // prevenir high optimization do kernel
    INIT_SHA512(H);
    #pragma unroll
    for(int i = 0; i<2048;i++){
        sha512_process(M, H); M[0]^=i; // prevenir high optimization do kernel ao mesmo tempo que testa somente o sha proccess block
    }
    out[gid] = H[0]; // prevenir high optimization do kernel
}

"""

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
print("OpenCL device:", queue.device.name)

prg = cl.Program(ctx, common_code + "\n" + sha512_code + "\n" + kernel_code).build()
k1 = prg.bench_comp1
mf = cl.mem_flags

def bench(kernel, n, local,runs=10):
    outbuf = cl.Buffer(ctx, mf.WRITE_ONLY, n * 8)
    ts = []
    for _ in range(runs):
        kernel.set_args(outbuf)
        ev = cl.enqueue_nd_range_kernel(queue, kernel, (n,), (local,))
        ev.wait()
        ts.append((ev.profile.end - ev.profile.start) * 1e-9)

    t = sum(ts) / len(ts)
    comp_ops_s = (n*2048) / t
    return t, comp_ops_s

# ----------------------------
# ----------------------------
N     = 100_000_000   
LOCAL = 64
N-=N%LOCAL     
RUNS  = 5

t1, comp1 = bench(k1, N, LOCAL)

print("\n==== COMPRESSÕES/SEG  ====")
print(f"1-bloco : N={N:,} | t={t1*1e3:.3f} ms | compress_ops/s={comp1:,.0f} ({comp1/1e6:.1f} MH/s)")


```

### Functions
- **`sha512_process(const ulong *message, ulong *H)`**: Performs a single SHA-512 compression on one 1024-bit block, updating the state `H`.
- **`sha512_hash_two_blocks_message(const ulong *message, ulong *H)`**: Initializes the state and compresses exactly two blocks (message[0..15] and message[16..31]).

**Note**: Padding and big-endian byte ordering must be handled by the caller. For HMAC, prepare the inner/outer prefixes accordingly to fit into two blocks.

## Performance Notes
- Tested for correctness against standard SHA-512 test vectors (e.g., via Python's `hashlib`).
- Achieves high GB/s throughput on modern GPUs due to unrolling and intrinsic optimizations.
- For benchmarks, run on your hardware; results vary by device (e.g., expect 10-100 GB/s on high-end GPUs for bulk 256-byte hashes).

## Author
- Bruno da Silva ([@ipsbruno3](https://github.com/ipsbruno3))

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
