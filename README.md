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


**OpenCL Device:** NVIDIA RTX PRO 6000 Blackwell Workstation Edition

| Configuration | Loops | Time (ms) | Compressions/s     | MH/s    |
|---------------|-------|-----------|--------------------|---------|
| 1-bloco      | 128   | 257.64    | 9,936,287,160      | 9936.3  |
| 2-blocos     | 64    | 252.86    | 10,124,336,982     | 10124.3 |

With these optimizations, we achieve impressive benchmarks of up to 10 billion compressions per second (in tests that exclude kernel launch overhead and rely solely on the global ID to compute the exact function result, avoiding any timing contamination from input/output buffer I/O).

**Benchmark Code**
```python
# bench_sha512_no_launch_compress.py
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
// 1 compressão por iteração
__kernel void bench_comp1(__global ulong *out, uint loops) {
  uint gid = get_global_id(0);

  __private ulong M[16];
  // gera bloco em registrador (barato e variável por gid)
  ulong x = ((ulong)gid) * 0x9e3779b97f4a7c15UL + 0xD1B54A32D192ED03UL;
  #pragma unroll
  for (int i=0;i<16;i++) {
    x = x * 0xbf58476d1ce4e5b9UL + 0x94d049bb133111ebUL;
    M[i] = x ^ ((ulong)i * 0x123456789abcdef0UL);
  }

  __private ulong H[8];
  INIT_SHA512(H);
  H[0] ^= (ulong)gid;

  for (uint r=0; r<loops; r++) {
    sha512_process(M, H);     // 1 compressão
    M[0] += H[0];             // dependência p/ não “sumir”
  }

  out[gid] = H[0];
}

// 2 compressões por iteração (2 blocos)
__kernel void bench_comp2(__global ulong *out, uint loops) {
  uint gid = get_global_id(0);

  __private ulong M[32];
  ulong x = ((ulong)gid) * 0x9e3779b97f4a7c15UL + 0x8a5cd789635d2dffUL;
  #pragma unroll
  for (int i=0;i<32;i++) {
    x = x * 0xbf58476d1ce4e5b9UL + 0x94d049bb133111ebUL;
    M[i] = x ^ ((ulong)i * 0x0f0e0d0c0b0a0908UL);
  }

  __private ulong H[8];
  INIT_SHA512(H);
  H[0] ^= (ulong)gid;

  for (uint r=0; r<loops; r++) {
    sha512_process(M, H);        // compressão 1
    sha512_process(M + 16, H);   // compressão 2
    M[0]  += H[0];
    M[16] ^= H[1];
  }

  out[gid] = H[0];
}
"""

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
print("OpenCL device:", queue.device.name)

prg = cl.Program(ctx, common_code + "\n" + sha512_code + "\n" + kernel_code).build()
k1 = prg.bench_comp1
k2 = prg.bench_comp2
mf = cl.mem_flags

def run_profile(kernel, N, local, loops, outbuf):
    kernel.set_args(outbuf, np.uint32(loops))
    ev = cl.enqueue_nd_range_kernel(queue, kernel, (N,), (local,))
    ev.wait()
    return (ev.profile.end - ev.profile.start) * 1e-9

def calibrate_loops(kernel, N, local, outbuf, target_ms=200.0, max_loops=1<<20):
    loops = 1
    # warmup curto
    for _ in range(2):
        run_profile(kernel, N, local, loops, outbuf)

    while True:
        t = run_profile(kernel, N, local, loops, outbuf)
        if (t * 1000.0) >= target_ms or loops >= max_loops:
            return loops, t
        loops <<= 1

def bench(kernel, N, local, blocks_per_iter, runs=7, target_ms=200.0):
    outbuf = cl.Buffer(ctx, mf.WRITE_ONLY, N * 8)  # 1 ulong por thread (mínimo)
    loops, _ = calibrate_loops(kernel, N, local, outbuf, target_ms=target_ms)
    ts = [run_profile(kernel, N, local, loops, outbuf) for _ in range(runs)]
    t = sum(ts) / len(ts)
    comp_ops_s = (N * loops * blocks_per_iter) / t
    return loops, t, comp_ops_s

# Ajustes seguros (sem estourar VRAM e sem TDR fácil)
N = 20_000_000   # 2^20 threads
LOCAL = 64

loops1, t1, comp1 = bench(k1, N, LOCAL, blocks_per_iter=1, runs=10, target_ms=200.0)
loops2, t2, comp2 = bench(k2, N, LOCAL, blocks_per_iter=2, runs=10, target_ms=200.0)

print("\n==== COMPRESSÕES/SEG (launch praticamente removido via loops no kernel) ====")
print(f"1-bloco : loops={loops1} | t={t1*1e3:.2f} ms | compress_ops/s={comp1:,.0f}  ({comp1/1e6:.1f} MH/s)")
print(f"2-blocos: loops={loops2} | t={t2*1e3:.2f} ms | compress_ops/s={comp2:,.0f}  ({comp2/1e6:.1f} MH/s)")

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
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (if not present, assume open-source for educational purposes).
