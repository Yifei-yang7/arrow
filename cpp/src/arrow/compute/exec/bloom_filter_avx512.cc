// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <immintrin.h>
#include "arrow/compute/exec/bloom_filter.h"
#include "arrow/util/bit_util.h"

namespace arrow {
namespace compute {

#if defined(ARROW_HAVE_AVX512)

inline __m512i BlockedBloomFilter::mask_avx512(__m512i hash) const {
  // AVX512 translation of mask() method
  //
  __m512i mask_id =
      _mm512_and_si512(hash, _mm512_set1_epi64(BloomFilterMasks::kNumMasks - 1));

  __m512i mask_byte_index = _mm512_srli_epi64(mask_id, 3);
  __m512i result = _mm512_i64gather_epi64(mask_byte_index, masks_.masks_, 1);
  __m512i mask_bit_in_byte_index = _mm512_and_si512(mask_id, _mm512_set1_epi64(7));
  result = _mm512_srlv_epi64(result, mask_bit_in_byte_index);
  result = _mm512_and_si512(result, _mm512_set1_epi64(BloomFilterMasks::kFullMask));

  __m512i rotation = _mm512_and_si512(
      _mm512_srli_epi64(hash, BloomFilterMasks::kLogNumMasks), _mm512_set1_epi64(63));

  result = _mm512_or_si512(
      _mm512_sllv_epi64(result, rotation),
      _mm512_srlv_epi64(result, _mm512_sub_epi64(_mm512_set1_epi64(64), rotation)));

  return result;
}

inline __m512i BlockedBloomFilter::block_id_avx512(__m512i hash) const {
  // AVX2 translation of block_id() method
  //
  __m512i result = _mm512_srli_epi64(hash, BloomFilterMasks::kLogNumMasks + 6);
  result = _mm512_and_si512(result, _mm512_set1_epi64(num_blocks_ - 1));
  return result;
}

int64_t BlockedBloomFilter::FindImp_avx512(int64_t num_rows, const uint64_t* hashes,
                                           uint8_t* result_bit_vector) const {
  constexpr int unroll = 8;

  for (int64_t i = 0; i < num_rows / unroll; ++i) {
    __m512i hash = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(hashes) + i);
    __m512i mask = mask_avx512(hash);
    __m512i block_id = block_id_avx512(hash);
    __m512i block = _mm512_i64gather_epi64(block_id, blocks_, sizeof(uint64_t));
    result_bit_vector[i] = _mm512_cmpeq_epi64_mask(_mm512_and_si512(block, mask), mask);
  }

  return num_rows - (num_rows % unroll);
}

int64_t BlockedBloomFilter::Find_avx512(int64_t num_rows, const uint64_t* hashes,
                                        uint8_t* result_bit_vector) const {
  return FindImp_avx512(num_rows, hashes, result_bit_vector);
}

int64_t BlockedBloomFilter::InsertImp_avx512(int64_t num_rows, const uint64_t* hashes) {
  constexpr int unroll = 8;

  for (int64_t i = 0; i < num_rows / unroll; ++i) {
    __m512i hash = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(hashes) + i);
    __m512i mask = mask_avx512(hash);
    __m512i block_id = block_id_avx512(hash);
    _mm512_i64scatter_epi64(blocks_, block_id, mask, sizeof(uint64_t));
  }

  return num_rows - (num_rows % unroll);
}

int64_t BlockedBloomFilter::Insert_avx512(int64_t num_rows, const uint64_t* hashes) {
  return InsertImp_avx512(num_rows, hashes);
}

#endif

}  // namespace compute
}  // namespace arrow
