import tqdm
import numpy as np

from checkers import DefaultRewardCalculator, CheckersEnvironment


def to_coordinates(idx):
    x = 7 - (idx - 1) // 4
    y = (idx - 1) % 4 * 2 + x % 2
    return x, y


class GameBuilder:
    def __init__(self, r_calculator=DefaultRewardCalculator()):
        self.env = CheckersEnvironment(american=True, r_calculator=r_calculator)
        self.states = []
        self.rewards = []
        self.actions = []
        self.current_team = 1

    def add_move(self, idx_from, idx_to):
        assert self.env.observation()[to_coordinates(idx_from)] != 0, 'Bad turn'
        assert self.env.observation()[to_coordinates(idx_to)] == 0, 'Bad turn'

        for i, move in enumerate(self.env.current_possible_actions_values()):
            if move[to_coordinates(idx_from)] == 0 and \
                    (move[to_coordinates(idx_to)] == self.current_team + 1 or
                     move[to_coordinates(idx_to)] == self.current_team + 3):
                new_s, r, is_done, _ = self.env.step(i)
                self.states.append(new_s)
                self.rewards.append(r)
                self.actions.append(i)
                self.current_team ^= 1
                return

        raise RuntimeError('bad turn: {}-{}'.format(idx_from, idx_to))

    def apply_result(self, result):
        if result[0] == 0.5:
            reward_fix = self.env.r_calculator.draw_reward()
        else:
            reward_fix = result * self.env.r_calculator.win_reward()

        self.rewards[-1] = reward_fix


def game_tokenizer(stream):
    game_data = []

    for line in stream:
        if not line.strip():
            yield game_data
            game_data = []
            continue
        game_data.append(line.strip())


def parse_result(result):
    if result.startswith('0-1'):
        return np.array([-1, 1])
    if result.startswith('1-0'):
        return np.array([1, -1])
    return np.array([0.5, 0.5])


def move_tokenizer(game):
    comment = False
    for line in game:
        if line[0] == '[':
            continue

        for token in line.split():
            if comment and not token[-1] == '}':
                continue
            elif comment:
                comment = False
                continue

            if token[0] == '{':
                if token[-1] != '}':
                    comment = True
                continue

            if token[-1] == '.':
                continue

            yield token


def parse_move(move):
    if '-' in move:
        return list(map(int, move.split('-')))

    return list(map(int, move.split('x')))


def is_result(token):
    return token == '0-1' or token == '1-0' or token == '1/2-1/2'


def parse_game(game):
    builder = GameBuilder()

    for move in move_tokenizer(game):
        if not is_result(move):
            ids = parse_move(move)
            builder.add_move(ids[0], ids[-1])
        else:
            result = parse_result(move)
            builder.apply_result(result)

    return builder


def parse_base(data, max_size=100, start=0, shuffle=False):
    games = []
    bad_games = []
    games_data = list(game_tokenizer(data))

    if shuffle:
        np.random.shuffle(games_data)

    for game in tqdm.tqdm(games_data[start:start + max_size]):
        try:
            games.append(parse_game(game))
        except Exception as e:
            print(e)
            print(game)
            bad_games.append(game)

    return games, bad_games
