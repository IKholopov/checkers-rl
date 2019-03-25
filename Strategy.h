#pragma once

#include <GameState.h>

struct IStrategy {
    virtual ~IStrategy() = default;
    virtual std::shared_ptr<GameState> ChooseNextState(std::shared_ptr<GameState> current) = 0;
};
