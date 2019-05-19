#include <Env.h>
#include <sstream>
#include <thread>

CheckersEnv::CheckersEnv(bool american) : game_(std::make_unique<GameLoop>(american))
{
}

std::shared_ptr<GameState> CheckersEnv::Step(std::shared_ptr<GameState> action)
{
    return game_->Step(action);
}

GameEnd CheckersEnv::Run(std::shared_ptr<IStrategy> white_strategy,
                                                                              std::shared_ptr<IStrategy> black_strategy,
                                                                              bool verbose)
{
    game_->SetWhiteStrategy(white_strategy);
    game_->SetBlackStrategy(black_strategy);

    auto result = game_->EvaluateGame(verbose);
    return GameEnd{result.first, result.second};
}

void CheckersEnv::Reset(bool american, std::shared_ptr<IStrategy> white_strategy, std::shared_ptr<IStrategy> black_strategy,
                        std::shared_ptr<GameState> start_state, int max_steps)
{
    // std::this_thread::sleep_for(std::chrono::milliseconds(500));
    game_ = std::make_unique<GameLoop>(american, white_strategy, black_strategy, start_state, max_steps);
}

std::string CheckersEnv::Render() const
{
    std::stringstream stream;
    CurrentState()->Dump(stream);
    return stream.str();
}

std::vector<CellStatus> CheckersEnv::StateValue() const
{
    return CurrentState()->StateValue();
}
