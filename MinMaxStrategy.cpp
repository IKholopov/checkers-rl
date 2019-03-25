#include <MinMaxStrategy.h>

std::shared_ptr<GameState> MinMaxStrategy::ChooseNextState(std::shared_ptr<GameState> current) {
    Score alpha = Score{EndResult::Losing}, beta = Score{EndResult::Win};

    Score bestScore = Score{EndResult::Losing};
    auto bestMove = current->Expand().front();
    for (const auto& move: current->Expand()) {
        auto score = lookupState(move, maxDepth - 1, alpha, beta, -1);
        if (score > bestScore) {
            bestMove = move;
            bestScore = score;
        }
    }

    return bestMove;
}

Score MinMaxStrategy::lookupState(std::shared_ptr<GameState> state, int depth, Score alpha,
                                  Score beta, int multiplier) {
    if (depth == 0) {
        return multiplier * scoringFunction(state);
    }

    if (state->IsTerminal()) {
        return multiplier * scoringFunction(state);
    }

    const auto& moves = state->Expand();

    Score bestScore = Score{EndResult::Losing};

    for (const auto& next: moves) {
        auto score = lookupState(next, depth - 1, alpha, beta, -multiplier);

        if (multiplier * score > bestScore) {
            bestScore = multiplier * score;
        }

        if (multiplier > 0) {
            alpha = std::max(alpha, score);
        } else {
            beta = std::min(beta, score);
        }

        if (beta < alpha) {
            break;
        }
    }

    return multiplier * bestScore;
}
