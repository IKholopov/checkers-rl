#include <StrategyFactory.h>

#include <RandomStrategy.h>
#include <MinMaxStrategy.h>
#include <MCSTStrategy.h>
#include <QNNStrategy.h>

static int CheckersRate(std::shared_ptr<const GameState> state) {
    int rate = 0;
    for (int x = 0; x < BoardSize; ++x) {
        for (int y = 0; y < BoardSize; ++y) {
            auto cell_status = state->At(x, y);
            if (cell_status != CellStatus::None && !IsQueen(cell_status) &&
                state->CurrentTeam == TeamOfCell(cell_status)) {
                rate++;
            } else if (cell_status != CellStatus::None && !IsQueen(cell_status)) {
                rate--;
            }
        }
    }

    return rate;
}

static int KingsRate(std::shared_ptr<const GameState> state) {
    int rate = 0;
    for (int x = 0; x < BoardSize; ++x) {
        for (int y = 0; y < BoardSize; ++y) {
            auto cell_status = state->At(x, y);
            if (cell_status != CellStatus::None && IsQueen(cell_status) &&
                state->CurrentTeam == TeamOfCell(cell_status)) {
                rate += 1;
            } else if (cell_status != CellStatus::None && IsQueen(cell_status)) {
                rate -= 1;
            }
        }
    }

    return rate;
}


namespace Strategy {

std::shared_ptr<IStrategy> MakeRandomStrategy(int seed)
{
    return std::make_shared<RandomStrategy>(seed);
}

std::shared_ptr<IStrategy> MakeMinMaxStrategy(int max_depth)
{
    return std::make_shared<MinMaxStrategy>(max_depth, [](std::shared_ptr<const GameState> state) -> Score {
        if (state->IsTerminal()) {
            return EndResult::Losing;
        }

        return CheckersRate(state) + KingsRate(state) * 10;
    });
}

template <Team team>
using WeightedMCSTBase = MCSTStrategy<WeightedCheckerMCST<team, std::mt19937> >;

template <Team team>
class WeightedMCST: public IStrategy {
public:
    explicit WeightedMCST(int max_iteration) : base_(randomizer_, max_iteration) {
    }

    virtual std::shared_ptr<GameState> ChooseNextState(std::shared_ptr<GameState> current) override {
        assert(current->CurrentTeam == team);
        auto chosen = base_.ChooseNextState(current);
        assert(chosen->CurrentTeam == Opponent(team));
        return chosen;
    }

private:
    std::mt19937 randomizer_;
    WeightedMCSTBase<team> base_;
};

std::shared_ptr<IStrategy> MakeMCSTStrategy(Team team, int max_iterations)
{
    if (team == Team::White) {
        return std::make_shared<WeightedMCST<Team::White> >(max_iterations);
    } else {
        return std::make_shared<WeightedMCST<Team::Black> >(max_iterations);
    }
}

std::shared_ptr<IStrategy> MakeQNNStrategy(Team team, const std::string& path)
{
    return std::make_shared<QnnStrategy>(path, team);
}

}
