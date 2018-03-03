// cudamatrix/cu-array.cc

// Copyright 2016  Brno University of Technology (author: Karel Vesely)
//           2017  Shiyin Kang


// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <vector>

#if HAVE_CUDA == 1
#include <cuda_runtime_api.h>
#endif

#include "base/timer.h"
#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-matrixdim.h"
#include "cudamatrix/cu-kernels.h"

#include "cudamatrix/cu-array.h"

namespace kaldi {

template<>
void CuArrayBase<int32>::Sequence(const int32 base) {
  if (dim_ == 0) return;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;

    dim3 dimBlock(CU1DBLOCK);
    dim3 dimGrid(n_blocks(Dim(), CU1DBLOCK));

    cuda_sequence(dimGrid, dimBlock, Data(), Dim(), base);
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    for (int32 i = 0; i < dim_; i++) {
      data_[i] = base + i;
    }
  }
}


template<>
void CuArrayBase<int32>::Set(const int32 &value) {
  if (dim_ == 0) return;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;

    dim3 dimBlock(CU2DBLOCK);
    dim3 dimGrid(n_blocks(Dim(), CU2DBLOCK));
    ::MatrixDim d = { 1, Dim(), Dim() };

    cuda_int32_set_const(dimGrid, dimBlock, data_, value, d);
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    for (int32 i = 0; i < dim_; i++) {
      data_[i] = value;
    }
  }
}


template<>
void CuArrayBase<int32>::Add(const int32 &value) {
  if (dim_ == 0) return;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;

    dim3 dimBlock(CU2DBLOCK);
    dim3 dimGrid(n_blocks(Dim(), CU2DBLOCK));
    ::MatrixDim d = { 1, Dim(), Dim() };

    cuda_int32_add(dimGrid, dimBlock, data_, value, d);
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    for (int32 i = 0; i < dim_; i++) {
      data_[i] += value;
    }
  }
}

/* void CuArrayBase<int32>::EqualElementMask(const CuArrayBase<int32> &mat, CuMatrix<T> *mask) const { */
/*   // Check the inputs: */
/*   KALDI_ASSERT(mat.Dim() == dim_); */
/*   KALDI_ASSERT(mask != NULL); */
/*   // Resizes the output matrix: */
/*   mask->Resize(dim_, 1, kSetZero); */

/* #if HAVE_CUDA == 1 */
/*   if (CuDevice::Instantiate().Enabled()) { */
/*     CuTimer tim; */
/*     dim3 dimGrid, dimBlock; */
/*     // CU1DBLOCK typedefed to 256 for vector operations */
/*     GetBlockSizesForSimpleMatrixOperation(dim_, 1, */
/*                                           &dimGrid, &dimBlock); */
/*     cuda_equal_element_mask(dimGrid, dimBlock, data_, mat.Data(), */
/*                             mask->Data(), dim_, mat.Stride(), */
/*                             mask->Stride()); */
/*     CU_SAFE_CALL(cudaGetLastError()); */

/*     CuDevice::Instantiate().AccuProfile(__func__, tim); */
/*   } else */
/* #endif */
/*   { */
/*     for (int32 r = 0; r < dim_; r++) { */
/*       (*mask)(r,0) = ((*this)(r) ==  mat(r) ? 1.0 : 0.0); */
/*     } */
/*   } */
/* } */

}  // namespace kaldi
