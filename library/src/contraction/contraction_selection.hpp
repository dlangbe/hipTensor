/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

#ifndef HIPTENSOR_CONTRACTION_HEURISTICS_HPP
#define HIPTENSOR_CONTRACTION_HEURISTICS_HPP

#include "contraction_solution.hpp"
#include <vector>

namespace hiptensor
{
    class ContractionSolution;
    struct PerfMetrics;

    namespace internal
    {
        template <typename T>
        __device__ inline double toDouble(T const& val);

        __device__ inline double maxDouble(double a, double b);

        inline double getEpsilon(hiptensorComputeType_t id);

        __device__ inline unsigned pcg_hash(unsigned input);
        
        template <unsigned range>
        __device__ inline float gen_random_float(unsigned input);

        template <typename DataType>
        __global__ void fillKernel(DataType* data, uint32_t elementSize);

        template <typename DataType>
        __host__ static inline void fillLaunchKernel(DataType* data, uint32_t elementSize);

        template <typename DataType>
        __global__ void fillValKernel(DataType* data, uint32_t elementSize, DataType value);

        template <typename DataType>
        __host__ static inline void
            fillValLaunchKernel(DataType* data, uint32_t elementSize, DataType value);

        __global__ static void
        maxReduceKernel(double* relativeError, uint32_t elements, uint32_t offset, uint32_t maxElements);

        template <typename DDataType>
        __global__ void compareEqualKernel(DDataType* deviceD,
                                        DDataType* hostD,
                                        double*    relativeError,
                                        uint32_t   elementsD);

        template <typename DDataType>
        std::pair<bool, double> compareEqualLaunchKernel(DDataType*             deviceD,
                                                        DDataType*             hostD,
                                                        std::size_t            elementsD,
                                                        hiptensorComputeType_t computeType,
                                                        double                 tolerance = 100.0);
    }

    hiptensorStatus_t bruteForceModel(ContractionSolution**                    winner,
                                      std::vector<ContractionSolution*> const& candidates,
                                      hipDataType                              typeA,
                                      std::vector<std::size_t> const&          a_ms_ks_lengths,
                                      std::vector<std::size_t> const&          a_ms_ks_strides,
                                      std::vector<int32_t> const&              a_ms_ks_modes,
                                      hipDataType                              typeB,
                                      std::vector<std::size_t> const&          b_ns_ks_lengths,
                                      std::vector<std::size_t> const&          b_ns_ks_strides,
                                      std::vector<int32_t> const&              b_ns_ks_modes,
                                      hipDataType                              typeD,
                                      std::vector<std::size_t> const&          d_ms_ns_lengths,
                                      std::vector<std::size_t> const&          d_ms_ns_strides,
                                      std::vector<int32_t> const&              d_ms_ns_modes,
                                      hipDataType                              typeE,
                                      std::vector<std::size_t> const&          e_ms_ns_lengths,
                                      std::vector<std::size_t> const&          e_ms_ns_strides,
                                      std::vector<int32_t> const&              e_ms_ns_modes,
                                      hiptensorComputeType_t                   computeType,
                                      const uint64_t                           workspaceSize);

    template <typename A,
              typename B,
              typename C,
              typename D,
              ContractionOpId_t ContractionOp,
              typename ComputeType>
    struct ActorCriticSelection
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hipDataType                                             typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         hipDataType                                             typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         hipDataType                                             typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         hipDataType                                             typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         const uint64_t                                          workspaceSize);
    };

    hiptensorStatus_t
        actorCriticModel(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hipDataType                                             typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         hipDataType                                             typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         hipDataType                                             typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         hipDataType                                             typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         hiptensorComputeType_t                                  computeType,
                         const uint64_t                                          workspaceSize);

} // namespace hiptensor

#endif // HIPTENSOR_CONTRACTION_HEURISTICS_HPP
