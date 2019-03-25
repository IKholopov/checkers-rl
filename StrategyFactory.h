#pragma once

#include <Strategy.h>

namespace Strategy {

std::shared_ptr<IStrategy> MakeRandomStrategy(int seed = -1);
std::shared_ptr<IStrategy> MakeMinMaxStrategy(int max_depth);
std::shared_ptr<IStrategy> MakeMCSTStrategy(Team team, int max_iterations);

}
