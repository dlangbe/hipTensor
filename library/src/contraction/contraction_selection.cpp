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

#ifndef CHECK_HIP_ALLOC
#define CHECK_HIP_ALLOC(status)               \
    if(status != hipSuccess)                  \
    {                                         \
        return HIPTENSOR_STATUS_ALLOC_FAILED; \
    }
#endif

#include "contraction_selection.hpp"
#include "performance.hpp"
#include "util.hpp"
#include "logger.hpp"

#include "contraction_cpu_reference.hpp"
#include "contraction_cpu_reference_impl.hpp"
#include "contraction_cpu_reference_instances.hpp"

namespace hiptensor
{
    namespace internal
    {
        template <typename T>
        __device__ inline double toDouble(T const& val)
        {
            // TODO: deprecate. Not needed anymore because bfloat16_t has float operator
            return static_cast<double>(val);
        }

        __device__ inline double maxDouble(double a, double b)
        {
            if(std::isinf(a) || std::isinf(b))
            {
                return std::numeric_limits<double>::infinity();
            }
            // Check for NaN
            else if(std::isnan(a) || std::isnan(b))
            {
                return std::numeric_limits<double>::signaling_NaN();
            }
            return a > b ? a : b;
        }

        inline double getEpsilon(hiptensorComputeType_t id)
        {
            auto toDouble = [](auto const& val) { return static_cast<double>(static_cast<float>(val)); };

            if(id == HIPTENSOR_COMPUTE_16F)
            {
                return toDouble(std::numeric_limits<_Float16>::epsilon());
            }
            else if(id == HIPTENSOR_COMPUTE_16BF)
            {
                return toDouble(std::numeric_limits<hip_bfloat16>::epsilon());
            }
            else if(id == HIPTENSOR_COMPUTE_32F)
            {
                return toDouble(std::numeric_limits<float>::epsilon());
            }
            else if(id == HIPTENSOR_COMPUTE_64F)
            {
                return toDouble(std::numeric_limits<double>::epsilon());
            }
            else if(id == HIPTENSOR_COMPUTE_8U)
            {
                return 0;
            }
            else if(id == HIPTENSOR_COMPUTE_8I)
            {
                return 0;
            }
            else if(id == HIPTENSOR_COMPUTE_32U)
            {
                return 0;
            }
            else if(id == HIPTENSOR_COMPUTE_32I)
            {
                return 0;
            }
            else if(id == HIPTENSOR_COMPUTE_C32F)
            {
                return toDouble(std::numeric_limits<float>::epsilon());
            }
            else if(id == HIPTENSOR_COMPUTE_C64F)
            {
                return toDouble(std::numeric_limits<double>::epsilon());
            }
            else
            {
        #if !NDEBUG
                std::cout << "Unhandled hiptensorComputeType_t: " << id << std::endl;
        #endif // !NDEBUG
                return 0;
            }
        }

        __device__ inline unsigned pcg_hash(unsigned input)
        {
            unsigned state = input * 747796405u + 2891336453u;
            unsigned word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
            return (word >> 22u) ^ word;
        }

        // gen random float in range [-range, range)
        template <unsigned range = 1>
        __device__ inline float gen_random_float(unsigned input)
        {
            return (static_cast<float>(pcg_hash(input)) / static_cast<float>(UINT_MAX) - 0.5f)
                * static_cast<float>(range) * 2;
        }

        // fill kernel for 'elementSize' elements
        template <typename DataType>
        __global__ void fillKernel(DataType* data, uint32_t elementSize)
        {
            uint32_t index = (blockIdx.x * blockDim.x + threadIdx.x);

            if(index < elementSize)
            {
                if constexpr(std::is_same_v<DataType, hipFloatComplex>)
                {
                    auto value  = gen_random_float(index);
                    data[index] = make_hipFloatComplex(value, value);
                }
                else if constexpr(std::is_same_v<DataType, hipDoubleComplex>)
                {
                    auto value  = static_cast<double>(gen_random_float(index));
                    data[index] = make_hipDoubleComplex(value, value);
                }
                else
                {
                    auto value  = gen_random_float(index);
                    data[index] = static_cast<DataType>(value);
                }
            }
        }

        template <typename DataType>
        __host__ static inline void fillLaunchKernel(DataType* data, uint32_t elementSize)
        {
            auto blockDim = dim3(1024, 1, 1);
            auto gridDim  = dim3(ceilDiv(elementSize, blockDim.x), 1, 1);
            hipLaunchKernelGGL((fillKernel<DataType>),
                            gridDim,
                            blockDim,
                            0,
                            0,
                            data,
                            elementSize);
        }

        // fill kernel wrapper for 'elementSize' elements with a specific value
        template <typename DataType>
        __global__ void fillValKernel(DataType* data, uint32_t elementSize, DataType value)
        {
            uint32_t index = (blockIdx.x * blockDim.x + threadIdx.x);

            if(index < elementSize)
            {
                data[index] = value;
            }
        }

        template <typename DataType>
        __host__ static inline void
            fillValLaunchKernel(DataType* data, uint32_t elementSize, DataType value)
        {
            auto blockDim = dim3(1024, 1, 1);
            auto gridDim  = dim3(ceilDiv(elementSize, blockDim.x), 1, 1);
            hipLaunchKernelGGL(
                (fillValKernel<DataType>), gridDim, blockDim, 0, 0, data, elementSize, value);
        }

        __global__ static void
        maxReduceKernel(double* relativeError, uint32_t elements, uint32_t offset, uint32_t maxElements)
        {
            double* localRelativeError = relativeError + (offset * elements * blockIdx.x);

            for(int i = elements >> 1; i > 0; i = i >> 1)
            {
                if(threadIdx.x < i && offset * (elements * blockIdx.x + threadIdx.x + i) < maxElements)
                {
                    localRelativeError[offset * threadIdx.x]
                        = maxDouble(localRelativeError[offset * threadIdx.x],
                                    localRelativeError[offset * (threadIdx.x + i)]);
                }
                __syncthreads();
            }
        }

        template <typename DDataType>
        __global__ void compareEqualKernel(DDataType* deviceD,
                                        DDataType* hostD,
                                        double*    relativeError,
                                        uint32_t   elementsD)
        {
            uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

            if(index < elementsD)
            {
                DDataType valDevice = deviceD[index];
                DDataType valHost   = hostD[index];

                auto numerator = fabs(toDouble(valDevice) - toDouble(valHost));
                auto divisor   = fabs(toDouble(valDevice)) + fabs(toDouble(valHost)) + 1.0;

                if(std::isinf(numerator) || std::isinf(divisor))
                {
                    relativeError[index] = std::numeric_limits<DDataType>::infinity();
                }
                else if(std::isnan(numerator) || std::isnan(divisor))
                {
                    relativeError[index] = std::numeric_limits<DDataType>::signaling_NaN();
                }
                else
                {
                    relativeError[index] = numerator / divisor;
                }
            }
        }

        template <typename DDataType>
        std::pair<bool, double> compareEqualLaunchKernel(DDataType*             deviceD,
                                                        DDataType*             hostD,
                                                        std::size_t            elementsD,
                                                        hiptensorComputeType_t computeType,
                                                        double                 tolerance)
        {
            auto blockDim = dim3(1024, 1, 1);
            auto gridDim  = dim3(ceilDiv(elementsD, blockDim.x), 1, 1);

            void* d_relativeError;
            double  maxRelativeError;
            auto elementSize = elementsD * sizeof(double);
            CHECK_HIP_ERROR(hipMalloc(&d_relativeError, elementSize));

            hipEvent_t syncEvent;
            CHECK_HIP_ERROR(hipEventCreate(&syncEvent));

            // Calculate the relative error for each element of Tensor D
            hipLaunchKernelGGL((compareEqualKernel<DDataType>),
                            gridDim,
                            blockDim,
                            0,
                            0,
                            deviceD,
                            hostD,
                            (double*)d_relativeError,
                            elementsD);
            CHECK_HIP_ERROR(hipEventRecord(syncEvent));
            CHECK_HIP_ERROR(hipEventSynchronize(syncEvent));

            // Determine the maximum relative error
            blockDim             = dim3(512, 1, 1);
            uint32_t maxElements = 1024;
            uint32_t offset      = 1;

            for(uint32_t i = elementsD; i > 1; i = ceilDiv(i, maxElements))
            {
                gridDim       = dim3(ceilDiv(i, maxElements), 1, 1);
                auto elements = i > maxElements ? maxElements : i;

                hipLaunchKernelGGL((maxReduceKernel),
                                gridDim,
                                blockDim,
                                0,
                                0,
                                (double*)d_relativeError,
                                elements,
                                offset,
                                elementsD);

                CHECK_HIP_ERROR(hipEventRecord(syncEvent));
                CHECK_HIP_ERROR(hipEventSynchronize(syncEvent));
                offset = offset * maxElements;
            }
            CHECK_HIP_ERROR(
                hipMemcpy(&maxRelativeError, d_relativeError, sizeof(double), hipMemcpyDeviceToHost));

            // Free allocated device memory
            CHECK_HIP_ERROR(hipFree(d_relativeError));

            bool retval = true;
            bool isNaN  = std::isnan(maxRelativeError);

            auto toDouble
                = [](DDataType const& val) { return static_cast<double>(static_cast<float>(val)); };

            auto eps = getEpsilon(computeType);
            if(isNaN)
            {
                retval           = false;
                maxRelativeError = std::numeric_limits<DDataType>::signaling_NaN();
            }
            else if(maxRelativeError > (eps * tolerance))
            {
                retval = false;
            }

            return std::make_pair(retval, maxRelativeError);
        }
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
                                      const uint64_t                           workspaceSize)
    {
        // Make sure that we calculate full element space incase strides are not packed.
        auto sizeA = elementsFromLengths(a_ms_ks_lengths) * hipDataTypeSize(typeA);
        auto sizeB = elementsFromLengths(b_ns_ks_lengths) * hipDataTypeSize(typeB);
        auto sizeD = 0;
        if(typeD != NONE_TYPE)
        {
            sizeD = elementsFromLengths(d_ms_ns_lengths) * hipDataTypeSize(typeD);
        }
        auto sizeE = elementsFromLengths(e_ms_ns_lengths) * hipDataTypeSize(typeE);

        void *A_d, *B_d, *D_d, *E_d, *reference, *wspace;
        void *A_h, *B_h, *D_h, *E_h;

        /*
         * `alpha` and `beta` are void pointer. hiptensor uses readVal to load the value of alpha.
         * ```
         * alphaF = hiptensor::readVal<float>(
         *      alpha, convertToComputeType(HipDataType_v<typename Traits::ComputeDataT>));
         * ```
         * Hence, the `alpha` and `bete` need to point to a ComputeData value
         */
        ScalarData alpha;
        ScalarData beta;
        if(computeType == HIPTENSOR_COMPUTE_C32F || computeType == HIPTENSOR_COMPUTE_C64F)
        {
            writeVal(&alpha, computeType, {computeType, 1.02, 1.03});
            writeVal(&beta, computeType, {computeType, 1.04, 1.05});
        }
        else
        {
            writeVal(&alpha, computeType, ScalarData(computeType, 1.02));
            writeVal(&beta, computeType, ScalarData(computeType, 1.03));
        }

        CHECK_HIP_ALLOC(hipMalloc(&A_d, sizeA));
        CHECK_HIP_ALLOC(hipMalloc(&B_d, sizeB));
        CHECK_HIP_ALLOC(hipMalloc(&D_d, sizeD));
        CHECK_HIP_ALLOC(hipMalloc(&E_d, sizeE));
        CHECK_HIP_ALLOC(hipMalloc(&reference, sizeE));
        CHECK_HIP_ALLOC(hipMalloc(&wspace, workspaceSize));

        A_h = operator new(sizeA);
        B_h = operator new(sizeB);
        D_h = operator new(sizeD);
        E_h = operator new(sizeE);

        std::string          best_op_name;
        ContractionSolution* bestSolution = nullptr;
        PerfMetrics          bestMetrics  = {
            0,
            "",
            0,
            0,
            0,
        };

        size_t elementsA  = elementsFromLengths(a_ms_ks_lengths);
        size_t elementsB  = elementsFromLengths(b_ns_ks_lengths);
        size_t elementsCD = elementsFromLengths(e_ms_ns_lengths);

        // Initialize tensor data
        if(typeA == HIP_R_16F && typeB == HIP_R_16F && typeE == HIP_R_16F)
        {
            // Initialize matrix data on device
            internal::fillLaunchKernel<_Float16>((_Float16*)A_d, elementsA);
            internal::fillLaunchKernel<_Float16>((_Float16*)B_d, elementsB);
            if(typeD == HIP_R_16F)
            {
                internal::fillLaunchKernel<_Float16>((_Float16*)D_d, elementsCD);
            }
            internal::fillValLaunchKernel<_Float16>((_Float16*)E_d,
                                            elementsCD,
                                            std::numeric_limits<_Float16>::signaling_NaN());
        }
        else if(typeA == HIP_R_16BF && typeB == HIP_R_16BF && typeE == HIP_R_16BF)
        {
            // Initialize matrix data on device
            internal::fillLaunchKernel<hip_bfloat16>((hip_bfloat16*)A_d, elementsA);
            internal::fillLaunchKernel<hip_bfloat16>((hip_bfloat16*)B_d, elementsB);
            if(typeD == HIP_R_16BF)
            {
                internal::fillLaunchKernel<hip_bfloat16>((hip_bfloat16*)D_d,
                                                elementsCD);
            }
            internal::fillValLaunchKernel<hip_bfloat16>(
                (hip_bfloat16*)E_d,
                elementsCD,
                std::numeric_limits<hip_bfloat16>::signaling_NaN());
        }
        else if(typeA == HIP_R_32F && typeB == HIP_R_32F && typeE == HIP_R_32F)
        {
            // Initialize matrix data on device
            internal::fillLaunchKernel<float>((float*)A_d, elementsA);
            internal::fillLaunchKernel<float>((float*)B_d, elementsB);
            if(typeD == HIP_R_32F)
            {
                internal::fillLaunchKernel<float>((float*)D_d, elementsCD);
            }
            internal::fillValLaunchKernel<float>((float*)E_d,
                                        elementsCD,
                                        std::numeric_limits<float>::signaling_NaN());
        }
        else if(typeA == HIP_R_64F && typeB == HIP_R_64F && typeE == HIP_R_64F)
        {
            // Initialize matrix data on device
            internal::fillLaunchKernel<double>((double*)A_d, elementsA);
            internal::fillLaunchKernel<double>((double*)B_d, elementsB);
            if(typeD == HIP_R_64F)
            {
                internal::fillLaunchKernel<double>((double*)D_d, elementsCD);
            }
            internal::fillValLaunchKernel<double>((double*)E_d,
                                        elementsCD,
                                        std::numeric_limits<double>::signaling_NaN());
        }
        else if(typeA == HIP_C_32F && typeB == HIP_C_32F && typeE == HIP_C_32F)
        {
            // Initialize matrix data on device
            internal::fillLaunchKernel<hipFloatComplex>((hipFloatComplex*)A_d,
                                                elementsA);
            internal::fillLaunchKernel<hipFloatComplex>((hipFloatComplex*)B_d,
                                                elementsB);
            if(typeD == HIP_C_32F)
            {
                internal::fillLaunchKernel<hipFloatComplex>((hipFloatComplex*)D_d,
                                                    elementsCD);
            }
            internal::fillValLaunchKernel<hipFloatComplex>(
                (hipFloatComplex*)E_d,
                elementsCD,
                std::numeric_limits<hipFloatComplex>::signaling_NaN());
        }
        else if(typeA == HIP_C_64F && typeB == HIP_C_64F && typeE == HIP_C_64F)
        {
            // Initialize matrix data on device
            internal::fillLaunchKernel<hipDoubleComplex>((hipDoubleComplex*)A_d,
                                                elementsA);
            internal::fillLaunchKernel<hipDoubleComplex>((hipDoubleComplex*)B_d,
                                                elementsB);
            if(typeD == HIP_C_64F)
            {
                internal::fillLaunchKernel<hipDoubleComplex>((hipDoubleComplex*)D_d,
                                                    elementsCD);
            }
            internal::fillValLaunchKernel<hipDoubleComplex>(
                (hipDoubleComplex*)E_d,
                elementsCD,
                std::numeric_limits<hipDoubleComplex>::signaling_NaN());
        }

        // Copy tensor data to host
        CHECK_HIP_ERROR(hipMemcpy(A_h, A_d, sizeA, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(B_h, B_d, sizeB, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(D_h, D_d, sizeD, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(E_h, E_d, sizeE, hipMemcpyDeviceToHost));

        // Compute reference
        auto& instances   = hiptensor::ContractionCpuReferenceInstances::instance();
        auto  cpuCandidates
            = (typeD == hiptensor::NONE_TYPE) ? instances->allSolutions().query(
                typeA, typeB, hiptensor::NONE_TYPE, typeE, computeType)
                            : instances->allSolutions().query(typeA, typeB, typeD, typeE, computeType);

        auto refCandidate      = cpuCandidates.solutions().begin()->second;
        auto [errorCode, time] = (*refCandidate)(&alpha,
                                                A_h,
                                                B_h,
                                                &beta,
                                                D_h,
                                                E_h,
                                                a_ms_ks_lengths,
                                                a_ms_ks_strides,
                                                a_ms_ks_modes,
                                                b_ns_ks_lengths,
                                                b_ns_ks_strides,
                                                b_ns_ks_modes,
                                                d_ms_ns_lengths,
                                                d_ms_ns_strides,
                                                d_ms_ns_modes,
                                                e_ms_ns_lengths,
                                                e_ms_ns_strides,
                                                e_ms_ns_modes,
                                                wspace,
                                                0);
        // Copy reference to device
        CHECK_HIP_ERROR(hipMemcpy(reference, E_h, sizeE, hipMemcpyHostToDevice));

        for(auto* solution : candidates)
        {
            auto [errorCode, time] = (*solution)(&alpha,
                                                 A_d,
                                                 B_d,
                                                 &beta,
                                                 D_d,
                                                 E_d,
                                                 a_ms_ks_lengths,
                                                 a_ms_ks_strides,
                                                 a_ms_ks_modes,
                                                 b_ns_ks_lengths,
                                                 b_ns_ks_strides,
                                                 b_ns_ks_modes,
                                                 d_ms_ns_lengths,
                                                 d_ms_ns_strides,
                                                 d_ms_ns_modes,
                                                 e_ms_ns_lengths,
                                                 e_ms_ns_strides,
                                                 e_ms_ns_modes,
                                                 wspace,
                                                 workspaceSize,
                                                 StreamConfig{nullptr, true});
            if(errorCode == HIPTENSOR_STATUS_SUCCESS && time > 0)
            {
                // Validate solution
                bool validationResult;
                double maxRelativeError;

                if(typeE == HIP_R_16F)
                {
                    std::tie(validationResult, maxRelativeError)
                        = internal::compareEqualLaunchKernel<_Float16>((_Float16*)E_d,
                                                            (_Float16*)reference,
                                                            elementsCD,
                                                            computeType);
                }
                else if(typeE == HIP_R_16BF)
                {
                    std::tie(validationResult, maxRelativeError)
                        = internal::compareEqualLaunchKernel<hip_bfloat16>(
                            (hip_bfloat16*)E_d,
                            (hip_bfloat16*)reference,
                            elementsCD,
                            computeType);
                }
                else if(typeE == HIP_R_32F || typeE == HIP_C_32F)
                {
                    std::tie(validationResult, maxRelativeError)
                        = internal::compareEqualLaunchKernel<float>((float*)E_d,
                                                        (float*)reference,
                                                        elementsCD,
                                                        computeType);
                }
                else if(typeE == HIP_R_64F || typeE == HIP_C_64F)
                {
                    std::tie(validationResult, maxRelativeError)
                        = internal::compareEqualLaunchKernel<double>((double*)E_d,
                                                        (double*)reference,
                                                        elementsCD,
                                                        computeType);
                }

                // Make sure to time the kernels
                int32_t m, n, k;
                std::tie(m, n, k) = solution->problemDims();
                auto flops        = std::size_t(2) * m * n * k;
                auto bytes        = solution->problemBytes();

                PerfMetrics metrics = {
                    solution->uid(), // id
                    solution->kernelName(), // name
                    time, // avg time
                    static_cast<float>(flops) / static_cast<float>(1.E9) / time, // tflops
                    static_cast<float>(bytes) / static_cast<float>(1.E6) / time // BW
                };

                // Log Kernel performances access
                if (!validationResult)
                {
                    char msg[256];
                    snprintf(msg,
                            sizeof(msg),
                            "KernelId: %lu, KernelName: %s",
                            solution->uid(),
                            solution->kernelName().c_str());

                    printf("%s\n", msg);
                    printf("Validated: %s, maxRelativeError: %.6f\n", 
                        (validationResult ? "yes" : "no"), maxRelativeError);
                }
                if (validationResult)
                {
                    if((metrics > bestMetrics))
                    {
                        bestSolution = solution;
                        bestMetrics  = metrics;
                    }
                }
                else
                {
                    return HIPTENSOR_STATUS_EXECUTION_FAILED;
                }
            }
        }

        CHECK_HIP_ALLOC(hipFree(A_d));
        CHECK_HIP_ALLOC(hipFree(B_d));
        CHECK_HIP_ALLOC(hipFree(D_d));
        CHECK_HIP_ALLOC(hipFree(E_d));
        CHECK_HIP_ALLOC(hipFree(reference));
        CHECK_HIP_ALLOC(hipFree(wspace));

        free(A_h);
        free(B_h);
        free(D_h);
        free(E_h);

        *winner = bestSolution;

        if(bestSolution == nullptr)
        {
            return HIPTENSOR_STATUS_EXECUTION_FAILED;
        }
        else
        {
            return HIPTENSOR_STATUS_SUCCESS;
        }
    }

    template <>
    struct ActorCriticSelection<_Float16,
                                _Float16,
                                _Float16,
                                _Float16,
                                ContractionOpId_t::SCALE,
                                float>
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
                         const uint64_t                                          workspaceSize)
        {
            int d1 = a_ms_ks_lengths[0];
            int d2 = a_ms_ks_lengths[1];
            int d3 = b_ns_ks_lengths[0];
            int d4 = b_ns_ks_lengths[1];
            int d5 = a_ms_ks_lengths[2];
            int d6 = a_ms_ks_lengths[3];

            size_t unique_id = 0;

            unique_id = 11124293857315312720ull;

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<_Float16,
                                _Float16,
                                _Float16,
                                _Float16,
                                ContractionOpId_t::BILINEAR,
                                float>
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
                         const uint64_t                                          workspaceSize)
        {
            int d1 = a_ms_ks_lengths[0];
            int d2 = a_ms_ks_lengths[1];
            int d3 = b_ns_ks_lengths[0];
            int d4 = b_ns_ks_lengths[1];
            int d5 = a_ms_ks_lengths[2];
            int d6 = a_ms_ks_lengths[3];

            size_t unique_id = 0;

            unique_id = 1953020431947874122ull;

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<hip_bfloat16,
                                hip_bfloat16,
                                hip_bfloat16,
                                hip_bfloat16,
                                ContractionOpId_t::SCALE,
                                float>
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
                         const uint64_t                                          workspaceSize)
        {
            int d1 = a_ms_ks_lengths[0];
            int d2 = a_ms_ks_lengths[1];
            int d3 = b_ns_ks_lengths[0];
            int d4 = b_ns_ks_lengths[1];
            int d5 = a_ms_ks_lengths[2];
            int d6 = a_ms_ks_lengths[3];

            size_t unique_id = 0;

            unique_id = 14895098881714635802ull;

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<hip_bfloat16,
                                hip_bfloat16,
                                hip_bfloat16,
                                hip_bfloat16,
                                ContractionOpId_t::BILINEAR,
                                float>
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
                         const uint64_t                                          workspaceSize)
        {
            int d1 = a_ms_ks_lengths[0];
            int d2 = a_ms_ks_lengths[1];
            int d3 = b_ns_ks_lengths[0];
            int d4 = b_ns_ks_lengths[1];
            int d5 = a_ms_ks_lengths[2];
            int d6 = a_ms_ks_lengths[3];

            size_t unique_id = 0;

            unique_id = 8517235228581081946ull;

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<float, float, float, float, ContractionOpId_t::SCALE, _Float16>
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
                         const uint64_t                                          workspaceSize)
        {
            int d1 = a_ms_ks_lengths[0];
            int d2 = a_ms_ks_lengths[1];
            int d3 = b_ns_ks_lengths[0];
            int d4 = b_ns_ks_lengths[1];
            int d5 = a_ms_ks_lengths[2];
            int d6 = a_ms_ks_lengths[3];

            size_t unique_id = 0;

            unique_id = 17313709378682913599ull;

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<float, float, float, float, ContractionOpId_t::BILINEAR, _Float16>
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
                         const uint64_t                                          workspaceSize)
        {
            int d1 = a_ms_ks_lengths[0];
            int d2 = a_ms_ks_lengths[1];
            int d3 = b_ns_ks_lengths[0];
            int d4 = b_ns_ks_lengths[1];
            int d5 = a_ms_ks_lengths[2];
            int d6 = a_ms_ks_lengths[3];

            size_t unique_id = 0;

            unique_id = 14397647188602189900ull;

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<float, float, float, float, ContractionOpId_t::SCALE, hip_bfloat16>
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
                         const uint64_t                                          workspaceSize)
        {
            int d1 = a_ms_ks_lengths[0];
            int d2 = a_ms_ks_lengths[1];
            int d3 = b_ns_ks_lengths[0];
            int d4 = b_ns_ks_lengths[1];
            int d5 = a_ms_ks_lengths[2];
            int d6 = a_ms_ks_lengths[3];

            size_t unique_id = 0;

            unique_id = 8339198051871565944ull;

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<float,
                                float,
                                float,
                                float,
                                ContractionOpId_t::BILINEAR,
                                hip_bfloat16>
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
                         const uint64_t                                          workspaceSize)
        {
            int d1 = a_ms_ks_lengths[0];
            int d2 = a_ms_ks_lengths[1];
            int d3 = b_ns_ks_lengths[0];
            int d4 = b_ns_ks_lengths[1];
            int d5 = a_ms_ks_lengths[2];
            int d6 = a_ms_ks_lengths[3];

            size_t unique_id = 0;

            unique_id = 2724417728984064737ull;

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<float, float, float, float, ContractionOpId_t::SCALE, float>
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
                         const uint64_t                                          workspaceSize)
        {
            int d1 = a_ms_ks_lengths[0];
            int d2 = a_ms_ks_lengths[1];
            int d3 = b_ns_ks_lengths[0];
            int d4 = b_ns_ks_lengths[1];
            int d5 = a_ms_ks_lengths[2];
            int d6 = a_ms_ks_lengths[3];

            size_t unique_id = 0;

            unique_id = 5943247903036531691ull;

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<float, float, float, float, ContractionOpId_t::BILINEAR, float>
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
                         const uint64_t                                          workspaceSize)
        {
            int d1 = a_ms_ks_lengths[0];
            int d2 = a_ms_ks_lengths[1];
            int d3 = b_ns_ks_lengths[0];
            int d4 = b_ns_ks_lengths[1];
            int d5 = a_ms_ks_lengths[2];
            int d6 = a_ms_ks_lengths[3];

            size_t unique_id = 0;

            unique_id = 17972447156160297755ull;

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<double, double, double, double, ContractionOpId_t::SCALE, float>
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
                         const uint64_t                                          workspaceSize)
        {

            int d1 = a_ms_ks_lengths[0];
            int d2 = a_ms_ks_lengths[1];
            int d3 = b_ns_ks_lengths[0];
            int d4 = b_ns_ks_lengths[1];
            int d5 = a_ms_ks_lengths[2];
            int d6 = a_ms_ks_lengths[3];

            size_t unique_id = 0;

            unique_id = 3893144338697524749ull;

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<double, double, double, double, ContractionOpId_t::BILINEAR, float>
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
                         const uint64_t                                          workspaceSize)
        {
            int d1 = a_ms_ks_lengths[0];
            int d2 = a_ms_ks_lengths[1];
            int d3 = b_ns_ks_lengths[0];
            int d4 = b_ns_ks_lengths[1];
            int d5 = a_ms_ks_lengths[2];
            int d6 = a_ms_ks_lengths[3];

            size_t unique_id = 0;
            unique_id        = 15165261158317928321ull;

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<double, double, double, double, ContractionOpId_t::SCALE, double>
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
                         const uint64_t                                          workspaceSize)
        {

            int d1 = a_ms_ks_lengths[0];
            int d2 = a_ms_ks_lengths[1];
            int d3 = b_ns_ks_lengths[0];
            int d4 = b_ns_ks_lengths[1];
            int d5 = a_ms_ks_lengths[2];
            int d6 = a_ms_ks_lengths[3];

            size_t unique_id = 0;

            unique_id = 14511729289005214097ull;

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<double, double, double, double, ContractionOpId_t::BILINEAR, double>
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
                         const uint64_t                                          workspaceSize)
        {
            int d1 = a_ms_ks_lengths[0];
            int d2 = a_ms_ks_lengths[1];
            int d3 = b_ns_ks_lengths[0];
            int d4 = b_ns_ks_lengths[1];
            int d5 = a_ms_ks_lengths[2];
            int d6 = a_ms_ks_lengths[3];

            size_t unique_id = 0;

            unique_id = 3636246152928348445ull;

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<hipFloatComplex,
                                hipFloatComplex,
                                hipFloatComplex,
                                hipFloatComplex,
                                ContractionOpId_t::SCALE_COMPLEX,
                                hipFloatComplex>
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
                         const uint64_t                                          workspaceSize)
        {

            int d1 = a_ms_ks_lengths[0];
            int d2 = a_ms_ks_lengths[1];
            int d3 = b_ns_ks_lengths[0];
            int d4 = b_ns_ks_lengths[1];
            int d5 = a_ms_ks_lengths[2];
            int d6 = a_ms_ks_lengths[3];

            size_t unique_id = 0;

            unique_id = 5711776907278244209ull;

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<hipFloatComplex,
                                hipFloatComplex,
                                hipFloatComplex,
                                hipFloatComplex,
                                ContractionOpId_t::BILINEAR_COMPLEX,
                                hipFloatComplex>
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
                         const uint64_t                                          workspaceSize)
        {
            int d1 = a_ms_ks_lengths[0];
            int d2 = a_ms_ks_lengths[1];
            int d3 = b_ns_ks_lengths[0];
            int d4 = b_ns_ks_lengths[1];
            int d5 = a_ms_ks_lengths[2];
            int d6 = a_ms_ks_lengths[3];

            size_t unique_id = 0;

            unique_id = 355777364055884033ull;

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<hipDoubleComplex,
                                hipDoubleComplex,
                                hipDoubleComplex,
                                hipDoubleComplex,
                                ContractionOpId_t::SCALE_COMPLEX,
                                hipDoubleComplex>
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
                         const uint64_t                                          workspaceSize)
        {

            int d1 = a_ms_ks_lengths[0];
            int d2 = a_ms_ks_lengths[1];
            int d3 = b_ns_ks_lengths[0];
            int d4 = b_ns_ks_lengths[1];
            int d5 = a_ms_ks_lengths[2];
            int d6 = a_ms_ks_lengths[3];

            size_t unique_id = 0;

            unique_id = 3085227716611397774ull;

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<hipDoubleComplex,
                                hipDoubleComplex,
                                hipDoubleComplex,
                                hipDoubleComplex,
                                ContractionOpId_t::BILINEAR_COMPLEX,
                                hipDoubleComplex>
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
                         const uint64_t                                          workspaceSize)
        {
            int d1 = a_ms_ks_lengths[0];
            int d2 = a_ms_ks_lengths[1];
            int d3 = b_ns_ks_lengths[0];
            int d4 = b_ns_ks_lengths[1];
            int d5 = a_ms_ks_lengths[2];
            int d6 = a_ms_ks_lengths[3];

            size_t unique_id = 0;

            unique_id = 2196983681630807584ull;

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
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
                         const uint64_t                                          workspaceSize)
    {
        if(typeA == HIP_R_16F && typeB == HIP_R_16F && typeD == NONE_TYPE && typeE == HIP_R_16F
           && computeType == HIPTENSOR_COMPUTE_32F)
        {
            return ActorCriticSelection<_Float16,
                                        _Float16,
                                        _Float16,
                                        _Float16,
                                        ContractionOpId_t::SCALE,
                                        float>::selectWinner(winner,
                                                             candidates,
                                                             typeA,
                                                             a_ms_ks_lengths,
                                                             a_ms_ks_strides,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             workspaceSize);
        }
        else if(typeA == HIP_R_16F && typeB == HIP_R_16F && typeD == HIP_R_16F && typeE == HIP_R_16F
                && computeType == HIPTENSOR_COMPUTE_32F)
        {
            return ActorCriticSelection<_Float16,
                                        _Float16,
                                        _Float16,
                                        _Float16,
                                        ContractionOpId_t::BILINEAR,
                                        float>::selectWinner(winner,
                                                             candidates,
                                                             typeA,
                                                             a_ms_ks_lengths,
                                                             a_ms_ks_strides,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             workspaceSize);
        }
        else if(typeA == HIP_R_16BF && typeB == HIP_R_16BF && typeD == NONE_TYPE
                && typeE == HIP_R_16BF && computeType == HIPTENSOR_COMPUTE_32F)
        {
            return ActorCriticSelection<hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        ContractionOpId_t::SCALE,
                                        float>::selectWinner(winner,
                                                             candidates,
                                                             typeA,
                                                             a_ms_ks_lengths,
                                                             a_ms_ks_strides,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             workspaceSize);
        }
        else if(typeA == HIP_R_16BF && typeB == HIP_R_16BF && typeD == HIP_R_16BF
                && typeE == HIP_R_16BF && computeType == HIPTENSOR_COMPUTE_32F)
        {
            return ActorCriticSelection<hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        ContractionOpId_t::BILINEAR,
                                        float>::selectWinner(winner,
                                                             candidates,
                                                             typeA,
                                                             a_ms_ks_lengths,
                                                             a_ms_ks_strides,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             workspaceSize);
        }
        else if(typeA == HIP_R_32F && typeB == HIP_R_32F && typeD == NONE_TYPE && typeE == HIP_R_32F
                && computeType == HIPTENSOR_COMPUTE_16F)
        {
            return ActorCriticSelection<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::SCALE,
                                        _Float16>::selectWinner(winner,
                                                                candidates,
                                                                typeA,
                                                                a_ms_ks_lengths,
                                                                a_ms_ks_strides,
                                                                typeB,
                                                                b_ns_ks_lengths,
                                                                b_ns_ks_strides,
                                                                typeD,
                                                                d_ms_ns_lengths,
                                                                d_ms_ns_strides,
                                                                typeE,
                                                                e_ms_ns_lengths,
                                                                e_ms_ns_strides,
                                                                workspaceSize);
        }
        else if(typeA == HIP_R_32F && typeB == HIP_R_32F && typeD == HIP_R_32F && typeE == HIP_R_32F
                && computeType == HIPTENSOR_COMPUTE_16F)
        {
            return ActorCriticSelection<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::BILINEAR,
                                        _Float16>::selectWinner(winner,
                                                                candidates,
                                                                typeA,
                                                                a_ms_ks_lengths,
                                                                a_ms_ks_strides,
                                                                typeB,
                                                                b_ns_ks_lengths,
                                                                b_ns_ks_strides,
                                                                typeD,
                                                                d_ms_ns_lengths,
                                                                d_ms_ns_strides,
                                                                typeE,
                                                                e_ms_ns_lengths,
                                                                e_ms_ns_strides,
                                                                workspaceSize);
        }
        else if(typeA == HIP_R_32F && typeB == HIP_R_32F && typeD == NONE_TYPE && typeE == HIP_R_32F
                && computeType == HIP_R_16BF)
        {
            return ActorCriticSelection<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::SCALE,
                                        hip_bfloat16>::selectWinner(winner,
                                                                    candidates,
                                                                    typeA,
                                                                    a_ms_ks_lengths,
                                                                    a_ms_ks_strides,
                                                                    typeB,
                                                                    b_ns_ks_lengths,
                                                                    b_ns_ks_strides,
                                                                    typeD,
                                                                    d_ms_ns_lengths,
                                                                    d_ms_ns_strides,
                                                                    typeE,
                                                                    e_ms_ns_lengths,
                                                                    e_ms_ns_strides,
                                                                    workspaceSize);
        }
        else if(typeA == HIP_R_32F && typeB == HIP_R_32F && typeD == HIP_R_32F && typeE == HIP_R_32F
                && computeType == HIP_R_16BF)
        {
            return ActorCriticSelection<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::BILINEAR,
                                        hip_bfloat16>::selectWinner(winner,
                                                                    candidates,
                                                                    typeA,
                                                                    a_ms_ks_lengths,
                                                                    a_ms_ks_strides,
                                                                    typeB,
                                                                    b_ns_ks_lengths,
                                                                    b_ns_ks_strides,
                                                                    typeD,
                                                                    d_ms_ns_lengths,
                                                                    d_ms_ns_strides,
                                                                    typeE,
                                                                    e_ms_ns_lengths,
                                                                    e_ms_ns_strides,
                                                                    workspaceSize);
        }
        else if(typeA == HIP_R_32F && typeB == HIP_R_32F && typeD == NONE_TYPE && typeE == HIP_R_32F
                && computeType == HIPTENSOR_COMPUTE_32F)
        {
            return ActorCriticSelection<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::SCALE,
                                        float>::selectWinner(winner,
                                                             candidates,
                                                             typeA,
                                                             a_ms_ks_lengths,
                                                             a_ms_ks_strides,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             workspaceSize);
        }
        else if(typeA == HIP_R_32F && typeB == HIP_R_32F && typeD == HIP_R_32F && typeE == HIP_R_32F
                && computeType == HIPTENSOR_COMPUTE_32F)
        {
            return ActorCriticSelection<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::BILINEAR,
                                        float>::selectWinner(winner,
                                                             candidates,
                                                             typeA,
                                                             a_ms_ks_lengths,
                                                             a_ms_ks_strides,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             workspaceSize);
        }
        else if(typeA == HIP_R_64F && typeB == HIP_R_64F && typeD == NONE_TYPE && typeE == HIP_R_64F
                && computeType == HIPTENSOR_COMPUTE_32F)
        {
            return ActorCriticSelection<double,
                                        double,
                                        double,
                                        double,
                                        ContractionOpId_t::SCALE,
                                        float>::selectWinner(winner,
                                                             candidates,
                                                             typeA,
                                                             a_ms_ks_lengths,
                                                             a_ms_ks_strides,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             workspaceSize);
        }
        else if(typeA == HIP_R_64F && typeB == HIP_R_64F && typeD == HIP_R_64F && typeE == HIP_R_64F
                && computeType == HIPTENSOR_COMPUTE_32F)
        {
            return ActorCriticSelection<double,
                                        double,
                                        double,
                                        double,
                                        ContractionOpId_t::BILINEAR,
                                        float>::selectWinner(winner,
                                                             candidates,
                                                             typeA,
                                                             a_ms_ks_lengths,
                                                             a_ms_ks_strides,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             workspaceSize);
        }
        else if(typeA == HIP_R_64F && typeB == HIP_R_64F && typeD == NONE_TYPE && typeE == HIP_R_64F
                && computeType == HIPTENSOR_COMPUTE_64F)
        {
            return ActorCriticSelection<double,
                                        double,
                                        double,
                                        double,
                                        ContractionOpId_t::SCALE,
                                        double>::selectWinner(winner,
                                                              candidates,
                                                              typeA,
                                                              a_ms_ks_lengths,
                                                              a_ms_ks_strides,
                                                              typeB,
                                                              b_ns_ks_lengths,
                                                              b_ns_ks_strides,
                                                              typeD,
                                                              d_ms_ns_lengths,
                                                              d_ms_ns_strides,
                                                              typeE,
                                                              e_ms_ns_lengths,
                                                              e_ms_ns_strides,
                                                              workspaceSize);
        }
        else if(typeA == HIP_R_64F && typeB == HIP_R_64F && typeD == HIP_R_64F && typeE == HIP_R_64F
                && computeType == HIPTENSOR_COMPUTE_64F)
        {
            return ActorCriticSelection<double,
                                        double,
                                        double,
                                        double,
                                        ContractionOpId_t::BILINEAR,
                                        double>::selectWinner(winner,
                                                              candidates,
                                                              typeA,
                                                              a_ms_ks_lengths,
                                                              a_ms_ks_strides,
                                                              typeB,
                                                              b_ns_ks_lengths,
                                                              b_ns_ks_strides,
                                                              typeD,
                                                              d_ms_ns_lengths,
                                                              d_ms_ns_strides,
                                                              typeE,
                                                              e_ms_ns_lengths,
                                                              e_ms_ns_strides,
                                                              workspaceSize);
        }
        else if(typeA == HIP_C_32F && typeB == HIP_C_32F && typeD == NONE_TYPE && typeE == HIP_C_32F
                && computeType == HIPTENSOR_COMPUTE_C32F)
        {
            return ActorCriticSelection<hipFloatComplex,
                                        hipFloatComplex,
                                        hipFloatComplex,
                                        hipFloatComplex,
                                        ContractionOpId_t::SCALE_COMPLEX,
                                        hipFloatComplex>::selectWinner(winner,
                                                                       candidates,
                                                                       typeA,
                                                                       a_ms_ks_lengths,
                                                                       a_ms_ks_strides,
                                                                       typeB,
                                                                       b_ns_ks_lengths,
                                                                       b_ns_ks_strides,
                                                                       typeD,
                                                                       d_ms_ns_lengths,
                                                                       d_ms_ns_strides,
                                                                       typeE,
                                                                       e_ms_ns_lengths,
                                                                       e_ms_ns_strides,
                                                                       workspaceSize);
        }
        else if(typeA == HIP_C_32F && typeB == HIP_C_32F && typeD == HIP_C_32F && typeE == HIP_C_32F
                && computeType == HIPTENSOR_COMPUTE_C32F)
        {
            return ActorCriticSelection<hipFloatComplex,
                                        hipFloatComplex,
                                        hipFloatComplex,
                                        hipFloatComplex,
                                        ContractionOpId_t::BILINEAR_COMPLEX,
                                        hipFloatComplex>::selectWinner(winner,
                                                                       candidates,
                                                                       typeA,
                                                                       a_ms_ks_lengths,
                                                                       a_ms_ks_strides,
                                                                       typeB,
                                                                       b_ns_ks_lengths,
                                                                       b_ns_ks_strides,
                                                                       typeD,
                                                                       d_ms_ns_lengths,
                                                                       d_ms_ns_strides,
                                                                       typeE,
                                                                       e_ms_ns_lengths,
                                                                       e_ms_ns_strides,
                                                                       workspaceSize);
        }
        else if(typeA == HIP_C_64F && typeB == HIP_C_64F && typeD == NONE_TYPE && typeE == HIP_C_64F
                && computeType == HIPTENSOR_COMPUTE_C64F)
        {
            return ActorCriticSelection<hipDoubleComplex,
                                        hipDoubleComplex,
                                        hipDoubleComplex,
                                        hipDoubleComplex,
                                        ContractionOpId_t::SCALE_COMPLEX,
                                        hipDoubleComplex>::selectWinner(winner,
                                                                        candidates,
                                                                        typeA,
                                                                        a_ms_ks_lengths,
                                                                        a_ms_ks_strides,
                                                                        typeB,
                                                                        b_ns_ks_lengths,
                                                                        b_ns_ks_strides,
                                                                        typeD,
                                                                        d_ms_ns_lengths,
                                                                        d_ms_ns_strides,
                                                                        typeE,
                                                                        e_ms_ns_lengths,
                                                                        e_ms_ns_strides,
                                                                        workspaceSize);
        }
        else if(typeA == HIP_C_64F && typeB == HIP_C_64F && typeD == HIP_C_64F && typeE == HIP_C_64F
                && computeType == HIPTENSOR_COMPUTE_C64F)
        {
            return ActorCriticSelection<hipDoubleComplex,
                                        hipDoubleComplex,
                                        hipDoubleComplex,
                                        hipDoubleComplex,
                                        ContractionOpId_t::BILINEAR_COMPLEX,
                                        hipDoubleComplex>::selectWinner(winner,
                                                                        candidates,
                                                                        typeA,
                                                                        a_ms_ks_lengths,
                                                                        a_ms_ks_strides,
                                                                        typeB,
                                                                        b_ns_ks_lengths,
                                                                        b_ns_ks_strides,
                                                                        typeD,
                                                                        d_ms_ns_lengths,
                                                                        d_ms_ns_strides,
                                                                        typeE,
                                                                        e_ms_ns_lengths,
                                                                        e_ms_ns_strides,
                                                                        workspaceSize);
        }
        return HIPTENSOR_STATUS_EXECUTION_FAILED;
    }
}
