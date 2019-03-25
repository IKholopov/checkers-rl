#include <Env.h>
#include <sstream>

CheckersEnv::CheckersEnv() : game_(std::make_unique<GameLoop>())
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

void CheckersEnv::Reset(std::shared_ptr<IStrategy> white_strategy, std::shared_ptr<IStrategy> black_strategy,
                        std::shared_ptr<GameState> start_state, int max_steps)
{
    game_ = std::make_unique<GameLoop>(white_strategy, black_strategy, start_state, max_steps);
}

std::string CheckersEnv::Render()
{
    std::stringstream stream;
    CurrentState()->Dump(stream);
    return stream.str();
}
