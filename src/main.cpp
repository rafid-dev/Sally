#include <iostream>
#include "network.hpp"
#include "types.hpp"

int main(int argc, char const *argv[])
{
    Network::Init();

    Board board;

    int32_t output = Network::Evaluate(board);
    std::cout << output << std::endl;

    return 0;
}
