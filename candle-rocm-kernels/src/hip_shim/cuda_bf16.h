// Shadows CUDA's <cuda_bf16.h> for the shared candle-kernels sources.
#pragma once

#include <hip/hip_bf16.h>

// Most of the bfloat16 API already matches between CUDA and HIP: the kernels'
// use of __float2bfloat16, __bfloat162float, __hmax, __hmin and arithmetic
// operators needs no bridging beyond this type alias.
typedef __hip_bfloat16 __nv_bfloat16;
typedef __hip_bfloat162 __nv_bfloat162;

// CUDA spells the NaN-propagating min/max with a `_nan` suffix. HIP's plain
// __hmax/__hmin on bfloat16 already propagate NaN, but the suffixed names do
// not exist, and compatibility.cuh only declares the __half variants (and only
// for __CUDA_ARCH__ < 800, which we compile past).
__device__ __forceinline__ __nv_bfloat16 __hmax_nan(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __hisnan(a) ? a : (__hisnan(b) ? b : __hmax(a, b));
}

__device__ __forceinline__ __nv_bfloat16 __hmin_nan(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __hisnan(a) ? a : (__hisnan(b) ? b : __hmin(a, b));
}

// CUDA's scalar float-to-bfloat16 conversion is spelled with a `_rn`
// (round-to-nearest) suffix; HIP's is not, and does not define the suffixed
// name at all. Both round to nearest, so this is a pure spelling alias.
__device__ __forceinline__ __nv_bfloat16 __float2bfloat16_rn(float f) {
    return __float2bfloat16(f);
}

// CUDA packs two floats into a bfloat162 with __floats2bfloat162_rn; HIP
// defines the fp16 equivalent (__floats2half2_rn) but not this one. A
// constrained template coexists with a future non-template overload of the
// same name — see hip_compat.h's atomicAdd shim for the rationale — so this
// stops being instantiated, rather than erroring as a redefinition, the day
// ROCm adds its own.
template <typename T, typename candle_hip::enable_if<
                          candle_hip::is_same<T, float>::value, int>::type = 0>
__device__ __forceinline__ __nv_bfloat162 __floats2bfloat162_rn(T a, T b) {
    return __nv_bfloat162{__float2bfloat16(a), __float2bfloat16(b)};
}
