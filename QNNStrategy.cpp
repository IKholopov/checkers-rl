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
#ifdef UI
    const int64_t index = *result.argmax().data<int64_t>();
#else  // Под виндой ругается матом, что не может слинковать.
    const long index = *result.argmax().data<long>();
#endif // UI
    assert(index >= 0 && index < actions.size());
    return actions[index];
}
