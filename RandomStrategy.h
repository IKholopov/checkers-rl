#pragma once

#include <Strategy.h>

#include <random>

class RandomStrategy : public IStrategy {
public:
    explicit RandomStrategy(int seed = -1) {
        if (seed == -1) {
            std::random_device rd;
            seed = rd();
        }
        generator = std::mt19937(seed);
    }

    std::shared_ptr<GameState> ChooseNextState(std::shared_ptr<GameState> current) override {
        assert(current != nullptr);
        const auto& moves = current->Expand();
        assert(!moves.empty());
        std::uniform_int_distribution<size_t> move(0, moves.size() - 1);

        return moves[move(generator)];
    }

private:
    std::mt19937 generator;
};

