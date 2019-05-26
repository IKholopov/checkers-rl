#include <A2CStrategy.h>
#include <assert.h>

std::shared_ptr<GameState> A2CStrategy::ChooseNextState(std::shared_ptr<GameState> current)
{
    assert(GetModule() != nullptr);
    const auto& actions = current->Expand();
    if (actions.size() == 0) {
        return current;
    }
    auto tensor = MakeBatch(current, actions).unsqueeze(0);
    auto lengths = MakeLengths(1);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor);
    inputs.push_back(lengths);
    auto net_result = GetModule()->forward(inputs).toTuple()->elements();
    auto log_probs = net_result[0].toTensor();
    auto value = net_result[0].toTensor();
    if (greedy) {
#ifdef UI
		const int64_t index = *log_probs.argmax().data<int64_t>();
#else
        const long index = *log_probs.argmax().data<long>();
#endif
        assert(index >= 0 && index < actions.size());
        return actions[index];
    } else {
#ifdef UI
		const int64_t index = *torch::multinomial( torch::exp( log_probs ), 1 ).detach().data<int64_t>();
#else
        const long index = *torch::multinomial(torch::exp(log_probs), 1).detach().data<long>();
#endif
        assert(index >= 0 && index < actions.size());
        return actions[index];
    }
}
