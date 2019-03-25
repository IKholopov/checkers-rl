import checkers_swig
import numpy as np

# A wrapper fixing some

class DefaultRewardCalculator:
    # returns two rewards - for white and black agent (w, b), w == -b
    def reward(self, old_state, new_state):
        team = old_state.CurrentTeam
        opponent = checkers_swig.Opponent(team)
        if new_state.IsTerminal():
            r = 10000
            return np.array([r, -r] if team == checkers_swig.Team_White else [-r, r])
        old_regular_opp = old_state.RegularChecks(opponent)
        new_regular_opp = new_state.RegularChecks(opponent)
        r = old_regular_opp - new_regular_opp
        old_queen_my = old_state.QueenChecks(team)
        new_queen_my = new_state.QueenChecks(team)
        old_queen_opp = old_state.QueenChecks(opponent)
        new_queen_opp = new_state.QueenChecks(opponent)
        r += (new_queen_my - old_queen_my + old_queen_opp - new_queen_opp) * 10
        return np.array([r, -r] if team == checkers_swig.Team_White else [-r, r])



class CheckersAgent:
    def __init__(self, r_calculator= DefaultRewardCalculator()):
        self.env = checkers_swig.CheckersEnv()
        self.r_calculator = r_calculator

    def step(self, action):
        old_state = self.env.CurrentState()
        assert action.CurrentTeam != old_state.CurrentTeam, 'Not a valid action'
        reward = self.r_calculator.reward(old_state, action)
        self.env.Step(action)
        new_state = self.env.CurrentState()
        if new_state != action:
            reward = self.r_calculator.reward(action, new_state)
        return new_state, reward, new_state.IsTerminal(), {}

    def reset(self, white_strategy=checkers_swig.istrateg_ptr(), black_strategy=checkers_swig.istrateg_ptr(), state=None,
                max_steps=10000):
        return self.env.Reset(white_strategy, black_strategy, state, max_steps)

    def render(self):
        print(self.env.Render())

    def is_done(self):
        return self.env.IsDone()

    def observation(self):
        return self.env.CurrentState()

    def possible_actions(self, state):
        return self.env.GetPossibleActions(state)

    def simulate(self, white_strategy, black_strategy, verbose=False):
        return self.env.Run(white_strategy, black_strategy, verbose)
