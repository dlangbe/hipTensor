/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2023 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifndef HIPTENSOR_CONTRACTION_TEST_HPP
#define HIPTENSOR_CONTRACTION_TEST_HPP

#include <gtest/gtest.h>

#include "contraction_common_test_params.hpp"

namespace hiptensor
{
    // struct ContractionTest
    //     : public ::testing::TestWithParam<ContractionCommonTestParams::ProblemParams,
    //                                                  typename ContractionCommonTestParams::TestComputeTypeT,
    //                                                  typename ContractionCommonTestParams::AlgorithmT,
    //                                                  typename ContractionCommonTestParams::OperatorT,
    //                                                  typename ContractionCommonTestParams::WorkSizePrefT,
    //                                                  typename ContractionCommonTestParams::LogLevelT,
    //                                                  typename ContractionCommonTestParams::LengthsT,
    //                                                  typename ContractionCommonTestParams::StridesT,
    //                                                  typename ContractionCommonTestParams::AlphaT,
    //                                                  typename ContractionCommonTestParams::BetaT>>

    struct ContractionTest
        : public ::testing::TestWithParam<ContractionCommonTestParams::ProblemParams>
    {
        // using Base = ::testing::TestWithParam<std::tuple<typename ContractionCommonTestParams::TestDataTypeT,
        //                                                  typename ContractionCommonTestParams::TestComputeTypeT,
        //                                                  typename ContractionCommonTestParams::AlgorithmT,
        //                                                  typename ContractionCommonTestParams::OperatorT,
        //                                                  typename ContractionCommonTestParams::WorkSizePrefT,
        //                                                  typename ContractionCommonTestParams::LogLevelT,
        //                                                  typename ContractionCommonTestParams::LengthsT,
        //                                                  typename ContractionCommonTestParams::StridesT,
        //                                                  typename ContractionCommonTestParams::AlphaT,
        //                                                  typename ContractionCommonTestParams::BetaT>>;

        using Base = ::testing::TestWithParam<ContractionCommonTestParams::ProblemParams>;
     
        void SetUp() override
        {
            // Construct ProblemParams from
            // incoming gtest parameterization
            auto param             = Base::GetParam();
            auto dataTypes         = std::get<0>(param);
            auto computeType       = std::get<1>(param);
            auto algorithm         = std::get<2>(param);
            auto op                = std::get<3>(param);
            auto workSizePrefrence = std::get<4>(param);
            auto logLevel          = std::get<5>(param);
            auto problemLength     = std::get<6>(param);
            auto problemStride     = std::get<7>(param);
            auto alpha             = std::get<8>(param);
            auto beta              = std::get<9>(param);

            std::cout << "SetUp\n\n\n";

            // // Cleanup previously used resources if data types change
            // static KernelI* sLastKernelRun = nullptr;
            // if(sLastKernelRun && sLastKernelRun->getResource() != kernel->getResource())
            // {
            //     sLastKernelRun->getResource()->reset();
            // }
            // sLastKernelRun = kernel.get();

            // ProblemParams params = {threadBlock, problemSize, passDirection};

            // // Walk through kernel workflow
            // kernel->setup(params);
        }

        virtual void RunKernel()
        {
            // // Construct ProblemParams from
            // // incoming gtest parameterization
            // auto param  = Base::GetParam();
            // auto kernel = std::get<0>(param);
            // kernel->exec();
            // kernel->validateResults();
            // kernel->reportResults();

            std::cout << "RunKernel\n\n\n";
        }

        virtual void Warmup()
        {
            // auto param  = Base::GetParam();
            // auto kernel = std::get<0>(param);
            // kernel->exec();
        }

        void TearDown() override
        {
            // // Construct ProblemParams from
            // // incoming gtest parameterization
            // auto param  = Base::GetParam();
            // auto kernel = std::get<0>(param);
            // kernel->tearDown();
        }
    };
    // pass enum template values through Base::<name>

} // namespace hiptensor

#endif // HIPTENSOR_CONTRACTION_TEST_HPP
