#include <assert.h>
#include <QNNStrategy.h>

std::shared_ptr<GameState> QnnStrategy::ChooseNextState(std::shared_ptr<GameState> current)
{
    assert(GetModule() != nullptr);
    const auto& actions = current->Expand();
    if (actions.size() == 0) {
        return current;
    }
    auto tensor = MakeBatch(current, actions);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor);
    auto result = GetModule()->forward(inputs).toTensor();
    const long index = *result.argmax().data<long>();
    assert(index >= 0 && index < actions.size());
    return actions[index];
}
