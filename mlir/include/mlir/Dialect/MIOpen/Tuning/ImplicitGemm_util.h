//===- GridwiseGemmParams.h - MLIR tuning parameter generation --------*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines MLIR tuning parameter generation
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MIOPEN_IMPLICIT_GEMM_UTIL_H
#define MLIR_DIALECT_MIOPEN_IMPLICIT_GEMM_UTIL_H

#include "mlir/Dialect/MIOpen/Tuning/ConvContext.h"

using namespace mlir;

class ImplicitGemmUtil {
public:
  // greatest common divisor, aka highest common factor
  template <typename T> static T gcd(T x, T y) {
    if (x == y || x == 0) {
      return y;
    } else if (y == 0) {
      return x;
    } else if (x > y) {
      return gcd(x - y, y);
    } else {
      return gcd(x, y - x);
    }
  }

  template <typename T, typename... Ys> static T gcd(T x, Ys... ys) {
    return gcd(x, gcd(ys...));
  }

  // least common multiple
  template <typename T> static T lcm(T x, T y) {
    if (x == 0 || y == 0) {
      return 0;
    } else {
      return (x * y) / gcd(x, y);
    }
  }

  template <typename T, typename... Ys> static T lcm(T x, Ys... ys) {
    return lcm(x, lcm(ys...));
  }

  template <typename T> inline static T integer_divide_ceil(T x, T y) {
    return (x + y - 1) / y;
  }

  template <typename T> static T integer_least_multiple(T x, T y) {
    return y * integer_divide_ceil(x, y);
  }

  template <int64_t L, int64_t H>
  inline static bool IsTwoPower(const int64_t v) {
    static_assert(L <= H, "L <= H");
    if (((v - 1) & v) != 0)
      return false;
    return L <= v && v <= H;
  }

  template <int64_t L, int64_t H>
  inline static bool PreviousTwoPower(int64_t &v) {
    static_assert((((L - 1) & L) == 0), "L is not power of 2");
    static_assert((((H - 1) & H) == 0), "H is not power of 2");
    assert((IsTwoPower<L, H>(v)));
    if (v == L) {
      v = H;
      return true;
    }
    v /= 2;
    return false;
  }

  constexpr static std::size_t get_lds_max_number_of_byte() { return 65536; }

  static LogicalResult IsValidBlockwiseGemmXdlops(const ConvolutionContext &ctx,
                                                  const int64_t GemmMPerBlock,
                                                  const int64_t GemmNPerBlock,
                                                  const int64_t GemmKPerBlock,
                                                  const int64_t GemmMPerWave,
                                                  const int64_t GemmNPerWave,
                                                  const int64_t GemmKPack);

  static LogicalResult IsValidGridGemmXdlops(const std::size_t GemmM,
                                             const std::size_t GemmN,
                                             const std::size_t GemmK);

  static int64_t GetEPackLength(const ConvolutionContext &ctx) {
    // Based on data type, Es are packed
    int EPACK = 1;
    if (ctx.IsF16()) // for fp16, either 2 or 4 Es could be packed
    {
      if (ctx.isXdlOp) // in xdlops, 4 fp16s are packed
        EPACK = 4;
      else // for fp16, either 2 or 4 Es could be packed in non-xdlops
           // scenarios.
        // EPACK = (C * Y * X % 32) == 0 ? 4 : 2;
        EPACK = 2;
    } else if (ctx.IsBF16()) // for bfp16, only 2 Es could be packed
    {
      EPACK = 2;
    }
    return EPACK;
  }

  static void obtainGemmADimKVectorizable(
      mlir::miopen::ConvOpType opType,
      llvm::StringMap<std::pair<size_t, int64_t>> &dimIndexVal,
      bool &input1GemmKVectorizable);

  static void obtainGemmBDimKVectorizable(
      mlir::miopen::ConvOpType opType,
      llvm::StringMap<std::pair<size_t, int64_t>> &dimIndexVal,
      bool &input2GemmKVectorizable);
};
#endif // MLIR_DIALECT_MIOPEN_IMPLICIT_GEMM_UTIL_H
