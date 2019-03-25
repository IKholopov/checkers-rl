#pragma once

#include <Strategy.h>

#include <variant>
#include <functional>


enum class EndResult {
    Win,
    Losing
};

using Score = std::variant<int, EndResult>;

inline bool operator < (const Score& first, const Score& second) {
    if (second == Score{EndResult::Win}) {
        return first != Score{EndResult::Win};
    }

    if (second == Score{EndResult::Losing} || first == Score{EndResult::Win}) {
        return false;
    }

    if (first == Score{EndResult::Losing}) {
        return second != Score{EndResult::Losing};
    }

    return std::get<int>(first) < std::get<int>(second);
}

inline bool operator>(const Score& first, const Score& second) {
    return first != second && !(first < second);
}

inline Score operator-(const Score& score) {
    if (score == Score{EndResult::Win}) {
        return EndResult::Losing;
    }

    if (score == Score{EndResult::Losing}) {
        return EndResult::Win;
    }

    return -std::get<int>(score);
}

inline Score operator*(int multiplier, const Score& score) {
    if (score == Score{EndResult::Win} || score == Score{EndResult::Losing}) {
        if (multiplier > 0) {
            return score;
        } else if (multiplier < 0) {
            return -score;
        } else {
            return 0;
        }
    }

    return multiplier * std::get<int>(score);
}

class MinMaxStrategy : public IStrategy {
public:
    MinMaxStrategy(int maxDepth_, std::function<Score(std::shared_ptr<GameState>)> scoringFunction_)
            : scoringFunction(std::move(scoringFunction_)), maxDepth(maxDepth_) {
    }

    std::shared_ptr<GameState> ChooseNextState(std::shared_ptr<GameState> current) override;

private:
    std::function<Score(std::shared_ptr<GameState>)> scoringFunction;
    int maxDepth;

    Score lookupState(std::shared_ptr<GameState> state, int depth, Score alpha, Score beta, int multiplier);
};
