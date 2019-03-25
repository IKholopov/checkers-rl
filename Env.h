#pragma once

#include <common.h>

#include <GameLoop.h>

#include <memory>
#include <iostream>
#include <unordered_map>

struct GameEnd {
    Winner winner;
    std::vector<std::shared_ptr<GameState>> story;
};

class CheckersEnv {
public:
    CheckersEnv();

    std::shared_ptr<GameState> Step(std::shared_ptr<GameState> action);
    std::vector<std::shared_ptr<GameState>> GetPossibleActions(std::shared_ptr<GameState> state) const {
        return state->Expand();
    }
    std::shared_ptr<GameState> CurrentState() const {
        return game_->CurrentState();
    }
    GameEnd Run(std::shared_ptr<IStrategy> white_strategy, std::shared_ptr<IStrategy> black_strategy,
                bool verbose = false);
    void Reset(std::shared_ptr<IStrategy> white_strategy=nullptr, std::shared_ptr<IStrategy> black_strategy=nullptr,
               std::shared_ptr<GameState> start_state=nullptr, int max_steps = 10000);
    std::string Render();
    bool IsDone() const {
        return game_->IsFinished();
    }

private:
    std::unique_ptr<GameLoop> game_;
};
