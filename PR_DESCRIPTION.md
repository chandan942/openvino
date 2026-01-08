## What I Fixed

Fixed a divide-by-zero crash in the GPU plugin that happens when batch size is 0.

## The Issue

In the `calculate_dims()` function, when handling `dnnl::memory::format_tag::ab` format, the code was dividing by the batch size without checking if it's zero. This caused the plugin to crash when someone tried to use a tensor with batch size 0.

## What I Changed

I added a simple check before the division - if batch size is 0, we now throw a clear error message instead of crashing. I also stored the batch value in a variable to make the code cleaner and avoid calling `batch()` multiple times.

**File changed:** `src/plugins/intel_gpu/src/graph/impls/onednn/utils.cpp`

**The fix:**
- Check if batch size is 0 before dividing
- Throw a user-friendly error: `"[GPU] Invalid batch size: batch size == 0 is not allowed"`
- Use a local variable to store batch value for better readability

Fixes #24243
