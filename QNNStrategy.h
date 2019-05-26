#pragma once

#include <memory>

#include <torch/script.h>

#include <NNStrategy.h>

class QnnStrategy : public NNStrategy {
public:
    QnnStrategy(const std::string& path, Team team) : NNStrategy(path, team) {
    }

    virtual std::shared_ptr<GameState> ChooseNextState(std::shared_ptr<GameState> current) override;
};
