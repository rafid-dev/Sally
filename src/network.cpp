#include "network.hpp"
#include <cstring>
#include <memory>

#define INCBIN_STYLE INCBIN_STYLE_CAMEL
#include "incbin/incbin.h"
INCBIN(EVAL, "./nn.net");

using namespace chess;

namespace Network
{
#if defined(__AVX__) || defined(__AVX2__)
    alignas(ALIGNMENT) std::array<int16_t, INPUT_SIZE * HIDDEN_SIZE> inputWeights;
    alignas(ALIGNMENT) std::array<int16_t, HIDDEN_SIZE> inputBias;
    alignas(ALIGNMENT) std::array<int16_t, HIDDEN_SIZE * 2> hiddenWeights;
    alignas(ALIGNMENT) std::array<int32_t, OUTPUT_SIZE> hiddenBias;
#else
    std::array<int16_t, INPUT_SIZE * HIDDEN_SIZE> inputWeights;
    std::array<int16_t, HIDDEN_SIZE> inputBias;
    std::array<int16_t, HIDDEN_SIZE * 2> hiddenWeights;
    std::array<int32_t, OUTPUT_SIZE> hiddenBias;
#endif

    int32_t Evaluate(chess::Board &board)
    {
        Color side = board.sideToMove();
        auto heap_allocation_weights = std::make_unique<Weights>();
        heap_allocation_weights->load();
        heap_allocation_weights->refresh(board);

        Weights &weights = *heap_allocation_weights;

#if defined(__AVX__) || defined(__AVX2__)

        avx_register_type_16 reluBias{};
        avx_register_type_32 res{};

        const auto acc_act = (avx_register_type_16 *)weights[side].data();
        const auto acc_nac = (avx_register_type_16 *)weights[~side].data();
        const auto wgt = (avx_register_type_16 *)(hiddenWeights.data());

        for (int i = 0; i < HIDDEN_SIZE / STRIDE_16_BIT; i++)
        {
            res = avx_add_epi32(res, avx_madd_epi16(avx_max_epi16(acc_act[i], reluBias), wgt[i]));
        }

        for (int i = 0; i < HIDDEN_SIZE / STRIDE_16_BIT; i++)
        {
            res = avx_add_epi32(res,
                                avx_madd_epi16(avx_max_epi16(acc_nac[i], reluBias), wgt[i + HIDDEN_SIZE / STRIDE_16_BIT]));
        }

        const auto outp = sumRegisterEpi32(res) + hiddenBias[0];
        return outp / INPUT_WEIGHT_MULTIPLIER / HIDDEN_WEIGHT_MULTIPLIER;

#else
        int32_t output = hiddenBias[0];

        for (int chunks = 0; chunks < HIDDEN_SIZE / 256; chunks++)
        {
            const int offset = chunks * 256;
            for (int i = 0; i < 256; i++)
            {
                output += relu(weights[side][i + offset]) * hiddenWeights[i + offset];
            }
        }

        for (int chunks = 0; chunks < HIDDEN_SIZE / 256; chunks++)
        {
            const int offset = chunks * 256;
            for (int i = 0; i < 256; i++)
            {
                output += relu(weights[~side][i + offset]) * hiddenWeights[HIDDEN_SIZE + i + offset];
            }
        }

        return output / INPUT_WEIGHT_MULTIPLIER / HIDDEN_WEIGHT_MULTIPLIER;

#endif
    }

    void Init()
    {
        uint64_t memoryIndex = 0;

        std::memcpy(inputWeights.data(), &gEVALData[memoryIndex], INPUT_SIZE * HIDDEN_SIZE * sizeof(int16_t));
        memoryIndex += INPUT_SIZE * HIDDEN_SIZE * sizeof(int16_t);

        std::memcpy(inputBias.data(), &gEVALData[memoryIndex], HIDDEN_SIZE * sizeof(int16_t));
        memoryIndex += HIDDEN_SIZE * sizeof(int16_t);

        std::memcpy(hiddenWeights.data(), &gEVALData[memoryIndex], HIDDEN_DSIZE * OUTPUT_SIZE * sizeof(int16_t));
        memoryIndex += HIDDEN_DSIZE * OUTPUT_SIZE * sizeof(int16_t);

        std::memcpy(hiddenBias.data(), &gEVALData[memoryIndex], OUTPUT_SIZE * sizeof(int32_t));
        memoryIndex += OUTPUT_SIZE * sizeof(int32_t);

#ifdef DEBUG
        std::cout << "Memory index: " << memoryIndex << std::endl;
        std::cout << "Size: " << gEVALSize << std::endl;
        std::cout << "Bias: " << hiddenBias[0] / INPUT_WEIGHT_MULTIPLIER / HIDDEN_WEIGHT_MULTIPLIER << std::endl;

        std::cout << std::endl;
#endif
    }
}