import checkers_swig
import argparse
from tqdm import tqdm
from a2c import CheckersEnvWrapper, make_strat
import numpy as np

def test(net_path, path):
    stream = checkers_swig.stringstream()
    print('play white greedy vs ' + path)

    stats = {'white': 0, 'black': 0, 'draw': 0}
    for i in tqdm(range(100)):
        qnn = checkers_swig.MakeA2CStrategy(checkers_swig.Team_White, net_path, True)
        opp = make_strat(path)
        env = CheckersEnvWrapper(opp)
        result = env.simulate(qnn, opp())
        if result.winner == 0:
            stats['white'] += 1
        elif result.winner == 1:
            stats['black'] += 1
        elif result.winner == 2:
            stats['draw'] += 1
    print(stats)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('net_path', type=str)
    parser.add_argument('path', type=str)
    args = parser.parse_args()
    test(args.net_path, args.path)

if __name__ == '__main__':
    main()
