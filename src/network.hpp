#pragma once

#include <cstdint>
#include "chess.hpp"

namespace Network
{
#define BUCKETS (4)
#define INPUT_SIZE (64 * 12 * BUCKETS)
#define HIDDEN_SIZE (768)
#define HIDDEN_DSIZE (HIDDEN_SIZE * 2)
#define OUTPUT_SIZE (1)
#define INPUT_WEIGHT_MULTIPLIER (32)
#define HIDDEN_WEIGHT_MULTIPLIER (128)

extern std::array<int16_t, INPUT_SIZE * HIDDEN_SIZE> inputWeights;
extern std::array<int16_t, HIDDEN_SIZE> inputBias;
extern std::array<int16_t, HIDDEN_SIZE * 2> hiddenWeights;
extern std::array<int32_t, OUTPUT_SIZE> hiddenBias;
#include "simd.h"

// clang-format off
constexpr int KING_BUCKET[64] {
    0, 0, 1, 1, 1, 1, 0, 0,
    0, 0, 1, 1, 1, 1, 0, 0,
    0, 0, 1, 1, 1, 1, 0, 0,
    0, 0, 1, 1, 1, 1, 0, 0,
    2, 2, 3, 3, 3, 3, 2, 2,
    2, 2, 3, 3, 3, 3, 2, 2,
    2, 2, 3, 3, 3, 3, 2, 2,
    2, 2, 3, 3, 3, 3, 2, 2,
};
// clang-format on

static inline int16_t relu(int16_t input) { return std::max(static_cast<int16_t>(0), input); }

static inline int kingSquareIndex(chess::Square kingSquare, chess::Color kingColor) {
    kingSquare = chess::Square((56 * uint8_t(kingColor)) ^ kingSquare);
    return KING_BUCKET[kingSquare];
}

static inline int index(chess::PieceType pieceType, chess::Color pieceColor, chess::Square square, chess::Color view,
                 chess::Square kingSquare) {
    const int ksIndex = kingSquareIndex(kingSquare, view);
    square = chess::Square(square ^ (56 * uint8_t(view)));
    square = chess::Square(square ^ (7 * !!(kingSquare & 0x4)));

    // clang-format off
    return square
           + uint8_t(pieceType) * 64
           + !(uint8_t(pieceColor) ^ uint8_t(view)) * 64 * 6 + ksIndex * 64 * 6 * 2;
    // clang-format on
}

struct Weights {
#if defined(__AVX__) || defined(__AVX2__)
    alignas(ALIGNMENT) std::array<int16_t, HIDDEN_SIZE> white;
    alignas(ALIGNMENT) std::array<int16_t, HIDDEN_SIZE> black;
#else
    std::array<int16_t, HIDDEN_SIZE> white;
    std::array<int16_t, HIDDEN_SIZE> black;
#endif
    std::array<int16_t, HIDDEN_SIZE> &operator[](chess::Color side) { return side == chess::Color::WHITE ? white : black; }
    std::array<int16_t, HIDDEN_SIZE> &operator[](bool side) { return side ? black : white; }

    inline void load() {
        std::copy(std::begin(inputBias), std::end(inputBias), std::begin(white));
        std::copy(std::begin(inputBias), std::end(inputBias), std::begin(black));
    }

    inline void update(chess::PieceType pieceType, chess::Color pieceColor, chess::Square square,
                           chess::Square kingSquare_White, chess::Square kingSquare_Black){

        Weights& weights = *this;
#if defined(__AVX__) || defined(__AVX2__)
        for (auto side : {chess::Color::WHITE, chess::Color::BLACK}) {
            const int input =
                index(pieceType, pieceColor, square, side, side == chess::Color::WHITE ? kingSquare_White : kingSquare_Black);

            const auto wgt = reinterpret_cast<avx_register_type_16 *>(inputWeights.data());
            const auto inp = reinterpret_cast<avx_register_type_16 *>(weights[side].data());
            const auto out = reinterpret_cast<avx_register_type_16 *>(weights[side].data());

            constexpr int blockSize = 4; // Adjust the block size for optimal cache usage


                for (int block = 0; block < HIDDEN_SIZE / (STRIDE_16_BIT * blockSize); block++) {
                    const int baseIdx = (input * HIDDEN_SIZE / STRIDE_16_BIT) + (block * blockSize);

                    avx_register_type_16 *outPtr = out + (block * blockSize);
                    const avx_register_type_16 *wgtPtr = wgt + baseIdx;
                    const avx_register_type_16 *inpPtr = inp + (block * blockSize);

                    avx_register_type_16 sum0 = avx_add_epi16(inpPtr[0], wgtPtr[0]);
                    avx_register_type_16 sum1 = avx_add_epi16(inpPtr[1], wgtPtr[1]);
                    avx_register_type_16 sum2 = avx_add_epi16(inpPtr[2], wgtPtr[2]);
                    avx_register_type_16 sum3 = avx_add_epi16(inpPtr[3], wgtPtr[3]);

                    for (int i = 4; i < blockSize; i++) {
                        sum0 = avx_add_epi16(sum0, wgtPtr[i]);
                        sum1 = avx_add_epi16(sum1, wgtPtr[i + blockSize]);
                        sum2 = avx_add_epi16(sum2, wgtPtr[i + blockSize * 2]);
                        sum3 = avx_add_epi16(sum3, wgtPtr[i + blockSize * 3]);
                    }

                    outPtr[0] = sum0;
                    outPtr[1] = sum1;
                    outPtr[2] = sum2;
                    outPtr[3] = sum3;
                }
        }
#else
        for (auto side : {chess::Color::WHITE, chess::Color::BLACK}) {
            const int input =
                index(pieceType, pieceColor, square, side, side == chess::Color::WHITE ? kingSquare_White : kingSquare_Black);
            for (int chunks = 0; chunks < HIDDEN_SIZE / 256; chunks++) {
                const int offset = chunks * 256;
                for (int i = offset; i < 256 + offset; i++) {
                    weights[side][i] += inputWeights[input * HIDDEN_SIZE + i];
                }
            }
        }
#endif
    }

    inline void refresh(chess::Board& board){
        chess::U64 pieces = board.all();

        const chess::Square kingSquare_White = board.kingSq(chess::Color::WHITE);
        const chess::Square kingSquare_Black = board.kingSq(chess::Color::BLACK);

        while (pieces) {
            chess::Square sq = chess::poplsb(pieces);
            chess::Piece pt = board.pieceAt(sq);

            update(typeOfPiece(pt), board.colorOfPiece(pt), sq, kingSquare_White, kingSquare_Black);
        }
    }
};

int32_t Evaluate(chess::Board &);
void Init();
}