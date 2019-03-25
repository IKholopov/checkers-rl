#pragma once
#include <Strategy.h>

class GameLoop {
public:
    GameLoop(std::shared_ptr<IStrategy> white_strategy=nullptr, std::shared_ptr<IStrategy> black_strategy=nullptr,
             std::shared_ptr<GameState> start_state=nullptr, int max_steps = 10000);

    std::pair<Winner, std::vector<std::shared_ptr<GameState> > > EvaluateGame(bool verbose=false);

    std::shared_ptr<GameState> Step(std::shared_ptr<GameState> state);

    void SetWhiteStrategy(std::shared_ptr<IStrategy> startegy);
    void SetBlackStrategy(std::shared_ptr<IStrategy> startegy);

    std::shared_ptr<GameState> CurrentState() const;
    bool IsFinished() const {
        return steps_ >= max_steps_ || state_story_.back()->IsTerminal();
    }
    bool IsDraw() const {
        return steps_ == max_steps_;
    }

private:
    int max_steps_ = 10000;
    int steps_ = 0;
    std::shared_ptr<IStrategy> white_strategy_;
    std::shared_ptr<IStrategy> black_strategy_;
    std::vector<std::shared_ptr<GameState>> state_story_;
};
