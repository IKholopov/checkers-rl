#pragma once

#include <memory>

#include <torch/script.h>

#include <Strategy.h>

class QnnStrategy : public IStrategy {
public:
    QnnStrategy(const std::string& path, Team team) : team(team), module(torch::jit::load(path)) {
    }

    virtual std::shared_ptr<GameState> ChooseNextState(std::shared_ptr<GameState> current) override;

private:
    std::shared_ptr<torch::jit::script::Module> module;
    Team team;

    float Normalized(CellStatus status);
    torch::Tensor MakeBatch(std::shared_ptr<GameState> state,
                            const std::vector<std::shared_ptr<GameState>>& actions);
    static int InversedIndex(int index, bool inverse);
};
