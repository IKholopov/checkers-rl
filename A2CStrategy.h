#pragma once
#include <NNStrategy.h>

class A2CStrategy : public NNStrategy {
public:
    A2CStrategy(const std::string& path, Team team, bool greedy) : NNStrategy(path, team), greedy(greedy) {
    }
    virtual std::shared_ptr<GameState> ChooseNextState(std::shared_ptr<GameState> current) override;

private:
    bool greedy = false;
};
