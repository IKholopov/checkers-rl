#pragma once
#include <torch/script.h>

#include <Strategy.h>

class NNStrategy : public IStrategy {
public:
    NNStrategy(const std::string& path, Team team) :
        team(team), module(torch::jit::load(path)) {
    }

protected:
    float Normalized(CellStatus status) const;
    torch::Tensor MakeBatch(std::shared_ptr<GameState> state,
            const std::vector<std::shared_ptr<GameState>>& actions);
    torch::Tensor MakeLengths(int length);
    std::shared_ptr<torch::jit::script::Module> GetModule() const
        { return module; }

private:
    std::shared_ptr<torch::jit::script::Module> module;
    Team team;
};
