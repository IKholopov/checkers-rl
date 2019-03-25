#pragma once

#include <CheckerMCST.h>
#include <Strategy.h>

#include <random>

template <class MCST>
class MCSTStrategy : public IStrategy {
public:
    explicit MCSTStrategy(std::mt19937& engine, int max_iteration=100)
        : generator(engine),
          strategy(generator, max_iteration) {
    }

    std::shared_ptr<GameState> ChooseNextState(std::shared_ptr<GameState> current) override {
        if (!strategy.Initialized()) {
            strategy.Initialize(current);
        } else {
            strategy.OpponentMove(current);
        }

        return strategy.GetMove(nullptr);
    }
private:
    std::mt19937& generator;
    MCSTPlayer<MCST> strategy;
};
