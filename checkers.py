import checkers_swig
import numpy as np

# A wrapper fixing some SWIG issues and calculating rewards

class DefaultRewardCalculator:
    # returns two rewards - for white and black agent (w, b), w == -b
    def reward(self, old_state, new_state):
        team = old_state.CurrentTeam
        opponent = checkers_swig.Opponent(team)
        if new_state.IsTerminal():
            r = self.win_reward()
            return np.array([r, -r] if team == checkers_swig.Team_White else [-r, r])
        old_regular_opp = old_state.RegularChecks(opponent)
        new_regular_opp = new_state.RegularChecks(opponent)
        r = old_regular_opp - new_regular_opp
        old_queen_my = old_state.QueenChecks(team)
        new_queen_my = new_state.QueenChecks(team)
        old_queen_opp = old_state.QueenChecks(opponent)
        new_queen_opp = new_state.QueenChecks(opponent)
        r += (new_queen_my - old_queen_my + old_queen_opp - new_queen_opp) * 4
        return np.array([r, -r] if team == checkers_swig.Team_White else [-r, r])

    def draw_reward(self):
        return np.array([-self.win_reward(), -self.win_reward()])

    def win_reward(self):
        return 100


class CheckersEnvironment:
    def __init__(self, american=False, r_calculator= DefaultRewardCalculator()):
        self.env = checkers_swig.CheckersEnv(american)
        self.r_calculator = r_calculator

    def step(self, action):
        old_state = self.env.CurrentState()
        action = self.possible_actions(old_state)[action]
        assert action.CurrentTeam != old_state.CurrentTeam, 'Not a valid action'
        reward = self.r_calculator.reward(old_state, action)
        self.env.Step(action)
        new_state = self.env.CurrentState()

        if self.env.IsDraw():
            return action, self.r_calculator.draw_reward(), True, {}

        if new_state != action:
            reward = self.r_calculator.reward(action, new_state)
        return self._observation(new_state.StateValue()), \
            reward, new_state.IsTerminal(), {}

    def reset(self, american=False, white_strategy=checkers_swig.istrateg_ptr(), black_strategy=checkers_swig.istrateg_ptr(), state=None,
                max_steps=10000):
        return self.env.Reset(american, white_strategy, black_strategy, state, max_steps)

    def render(self, numpy=False):
        if numpy:
            return self._observation(self.env.StateValue())
        print(self.env.Render())

    def _observation(self, state_value):
        return np.array(state_value).reshape((checkers_swig.BoardSize, checkers_swig.BoardSize))

    def is_done(self):
        return self.env.IsDone()

    def observation(self):
        return self._observation(self.env.CurrentState().StateValue())

    def possible_actions(self, state):
        return self.env.GetPossibleActions(state)

    def current_possible_actions_values(self):
        return [self._observation(state.StateValue()) for state in self.env.GetPossibleActions(self.env.CurrentState())]

    def simulate(self, white_strategy, black_strategy, verbose=False):
        return self.env.Run(white_strategy, black_strategy, verbose)
