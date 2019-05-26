#include <NNStrategy.h>
#include <assert.h>

static int InversedIndex(int index, bool inverse)
{
    if (!inverse) {
        return index;
    }
    return BoardCells - 1 - index;
}

float NNStrategy::Normalized(CellStatus status) const
{
    switch (status) {
        case CellStatus::None:
            return 0.0f;
        case CellStatus::Black:
            return team == Team::White ? 1.0f : -1.0f;
        case CellStatus::White:
            return team == Team::White ? -1.0f : 1.0f;
        case CellStatus::BlackQueen:
            return team == Team::White ? 2.0f : -2.0f;
        case CellStatus::WhiteQueen:
            return team == Team::White ? -2.0f : 2.0f;
        default:
            assert(false);
            return 0.0f;
    }
}

torch::Tensor NNStrategy::MakeBatch(std::shared_ptr<GameState> state,
                                    const std::vector<std::shared_ptr<GameState> >& actions)
{
    const int size = actions.size() * BoardCells * 2;
    std::vector<float> inputBlob;
    inputBlob.reserve(size);
    for (int i = 0; i < actions.size(); ++i) {
        for (int j = 0; j < BoardCells; ++j) {
            inputBlob.push_back(Normalized(state->State[InversedIndex(j, team == Team::Black)]));
        }
        for (int j = 0; j < BoardCells; ++j) {
            inputBlob.push_back(Normalized(actions[i]->State[InversedIndex(j, team == Team::Black)]));
        }
    }
    torch::Tensor tensor = torch::empty({static_cast<int>(actions.size()), 2, BoardSize, BoardSize});
    for (int i = 0; i < size; ++i) {
        *(tensor.data<float>() + i) = inputBlob[i];
    }
    return tensor;
}

torch::Tensor NNStrategy::MakeLengths(int length)
{
    torch::Tensor tensor = torch::empty({1}, torch::dtype<int64_t>());
    *tensor.data<int64_t>() = length;
    return tensor;
}
