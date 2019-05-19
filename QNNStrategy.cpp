#include <assert.h>
#include <QNNStrategy.h>

float QnnStrategy::Normalized(CellStatus status) {
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

torch::Tensor QnnStrategy::MakeBatch(std::shared_ptr<GameState> state, const std::vector<std::shared_ptr<GameState>>& actions) {
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

int QnnStrategy::InversedIndex(int index, bool inverse)
{
    if (!inverse) {
        return index;
    }
    return BoardCells - 1 - index;
}

std::shared_ptr<GameState> QnnStrategy::ChooseNextState(std::shared_ptr<GameState> current)
{
    assert(module != nullptr);
    const auto& actions = current->Expand();
    if (actions.size() == 0) {
        return current;
    }
    auto tensor = MakeBatch(current, actions);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor);
    auto result = module->forward(inputs).toTensor();
    const long index = *result.argmax().data<long>();
    assert(index >= 0 && index < actions.size());
    return actions[index];
}
