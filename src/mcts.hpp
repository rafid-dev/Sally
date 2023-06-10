#pragma once

#include "network.hpp"
#include "types.hpp"

#include <unordered_set>
#include <random>
#include <algorithm>

constexpr int INF_BOUND = 30000;

struct Node
{
    Board state;
    GameResult result;
    int32_t evaluation;
    int32_t Q; // Reward
    uint32_t N; // Total visits

    std::vector<Node> children;

    inline Node &find_random_child()
    {
        size_t random_index = std::rand() % children.size();

        return children[random_index];
    }

    bool operator==(const Node& rhs){
        return state.hash() == rhs.state.hash();
    }

    inline bool terminal_node()
    {
        return (!children.size()) ? true : false;
    }
};

class MCTS
{
    std::vector<Node> children;

    double exploration_weight = 1.0;

    bool node_exists(const Node &node)
    {
        return std::find(children.begin(), children.end(), node) != children.end();
    }

    Node& find_node(Node& node){
        return children.at(std::find(children.begin(), children.end(), node) - children.end());
    }

    Node &choose(Node &node)
    {
        if (node.terminal_node())
        {
            throw std::runtime_error("Choose called on terminal node");
        }

        if (!node_exists(node)){
            return node.find_random_child();
        }
    }
};