#include <GameLoop.h>

GameLoop::GameLoop(bool american, std::shared_ptr<IStrategy> white_strategy, std::shared_ptr<IStrategy> black_strategy, std::shared_ptr<GameState> start_state, int max_steps)
    : max_steps_(max_steps),
      white_strategy_(std::move(white_strategy)),
      black_strategy_(std::move(black_strategy)) {
    if (start_state != nullptr) {
        state_story_.emplace_back(start_state);
    } else {
        state_story_.emplace_back(std::make_shared<GameState>(american));
    }
}

std::pair<Winner, std::vector<std::shared_ptr<GameState> > > GameLoop::EvaluateGame(bool verbose) {
    if (white_strategy_ == nullptr) {
        throw std::runtime_error("White strategy not set");
    }
    if (black_strategy_ == nullptr) {
        throw std::runtime_error("Black strategy not set");
    }
    auto last_state = state_story_.back();
    while (!last_state->IsTerminal() && steps_ < max_steps_) {
        if (verbose) {
            last_state->Dump(std::cerr);
        }
        switch (last_state->CurrentTeam) {
        case Team::Black: {
            state_story_.emplace_back(black_strategy_->ChooseNextState(last_state));
            break;
        }
        case Team::White: {
            state_story_.emplace_back(white_strategy_->ChooseNextState(last_state));
            break;
        }
        case Team::None: {
            assert(false);
        }
        }
        last_state = state_story_.back();
    }


    Winner winner =
            last_state->CurrentTeam == Team::White ? Winner::Black : Winner::White;
    if (steps_ == max_steps_ && !last_state->IsTerminal()) {
        winner = Winner::Draw;
    }

    return {winner, state_story_};
}

std::shared_ptr<GameState> GameLoop::Step(std::shared_ptr<GameState> state) {
    ++steps_;
    auto last_state = state_story_.back();
    const auto& expansion = last_state->Expand();
    bool found = false;
    for (const auto& exp : expansion) {
        if (*state == *exp) {
            found = true;
            break;
        }
    }
    if (!found) {
        throw std::runtime_error("invalid state passed into GameLoop::Step()");
    }

    state_story_.push_back(state);
    if (IsFinished()) {
        return state_story_.back();
    }
    switch (state->CurrentTeam) {
        case Team::White:
            if (white_strategy_ != nullptr) {
                state_story_.push_back(white_strategy_->ChooseNextState(state));
            }
            break;
        case Team::Black:
            if (black_strategy_ != nullptr) {
                state_story_.push_back(black_strategy_->ChooseNextState(state));
            }
            break;
        case Team::None:
            assert(false);
            break;
    }

    return state_story_.back();
}

void GameLoop::SetWhiteStrategy(std::shared_ptr<IStrategy> startegy) {
    white_strategy_ = startegy;
}

void GameLoop::SetBlackStrategy(std::shared_ptr<IStrategy> startegy) {
    black_strategy_ = startegy;
}

std::shared_ptr<GameState> GameLoop::CurrentState() const {
    assert(!state_story_.empty());
    return state_story_.back();
}
