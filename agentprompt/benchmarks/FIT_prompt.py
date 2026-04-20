FIT_TRITON_PROMPT = """Generate a Triton kernel optimized for {target_gpu} GPU for

{definition}

Triton Version: 3.3.1

Requirements:
- Write clean, efficient Triton code optimized for {target_gpu} architecture
- Use modern Triton syntax with proper grid computation and language features
- Include necessary imports (torch, triton, triton.language as tl)
- Implement the exact functionality described in the specification
- The reference code provides the mathematical specification but is unoptimized - your Triton implementation should match its computational accuracy while delivering high performance
- Use the definition's tensor shapes, dtypes, and axes information to guide memory access patterns and optimization strategies
- Optimize for {target_gpu} GPU characteristics (memory hierarchy, compute units, etc.)

The wrapper function MUST handle complete device management:
- Move CPU tensors to GPU if needed (use .cuda() when torch.cuda.is_available())
- Raise clear errors if CUDA is not available for GPU tensors
- Call the triton kernel with GPU tensors
- Move results back to original device of input tensors
- Handle both args and kwargs properly
- Preserve original tensor devices and restore them for outputs

IMPORTANT: Use only valid Python/Triton syntax:
- NO hexadecimal float literals (0x1.234p5) - use decimal equivalents
- NO C/CUDA specific syntax - this is Python/Triton code
- All code must be valid Python that passes ast.parse()

- Expose a "run" entry point function that can be called to execute the kernel
- Return only the code, no explanations or markdown formatting

Generate complete, runnable code only - no framework will add device handling wrapper code.

Generate the implementation:"""

TRITON_OPTIMIZATION_PROMPT = """You are optimizing a Triton kernel for {target_gpu} GPU. The current implementation has issues that need to be fixed.

Original Specification:
{definition}

Current Implementation Status:
{trace_logs}

Current Implementation:
{current_code}

Optimization Strategy:
1. ENSURE CORRECTNESS: If there are compile errors, runtime errors, or incorrect outputs, focus entirely on fixing these issues
   - Analyze compilation errors and fix syntax/API usage
   - Fix runtime errors like shape mismatches, memory access violations
   - Ensure numerical correctness matches the reference implementation

2. OPTIMIZE PERFORMANCE: if the current kernel is functionally correct, focus on performance optimizations
   - Optimize memory access patterns for {target_gpu}
   - Tune block sizes and grid dimensions
   - Use appropriate Triton language features for vectorization
   - Minimize global memory transactions

Requirements for the optimized implementation:
- Write clean, efficient Triton code optimized for {target_gpu} architecture
- Use modern Triton syntax with proper grid computation and language features
- Include necessary imports (torch, triton, triton.language as tl)
- Fix all identified issues from the feedback
- Maintain or improve computational accuracy
- Preserve the same function signature and device handling as specified

The wrapper function MUST handle complete device management:
- Move CPU tensors to GPU if needed (use .cuda() when torch.cuda.is_available())
- Raise clear errors if CUDA is not available for GPU tensors
- Call the triton kernel with GPU tensors
- Move results back to original device of input tensors
- Handle both args and kwargs properly
- Preserve original tensor devices and restore them for outputs

IMPORTANT: Use only valid Python/Triton syntax:
- NO hexadecimal float literals (0x1.234p5) - use decimal equivalents
- NO C/CUDA specific syntax - this is Python/Triton code
- All code must be valid Python that passes ast.parse()

- Expose a "run" entry point function that can be called to execute the kernel
- Return only the improved code, no explanations or markdown formatting

Generate the corrected and optimized implementation:"""