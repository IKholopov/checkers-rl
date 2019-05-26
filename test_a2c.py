import checkers_swig
from tqdm import tqdm

net_path='a2c_rand_minmax.trpt'

stream = checkers_swig.stringstream()
print('play white greedy')
random = checkers_swig.MakeRandomStrategy(100)
stats = {'white': 0, 'black': 0, 'draw': 0}
for i in tqdm(range(1000)):
    qnn_1 = checkers_swig.MakeA2CStrategy(checkers_swig.Team_White, net_path, True)
    env = checkers_swig.CheckersEnv(True)
    result = env.Run(qnn_1, random)
    if result.winner == 0:
        stats['white'] += 1
    elif result.winner == 1:
        stats['black'] += 1
    elif result.winner == 2:
        stats['draw'] += 1
print(stats)

print('play black greedy')
stats = {'white': 0, 'black': 0, 'draw': 0}
for i in tqdm(range(1000)):
    qnn_2 = checkers_swig.MakeA2CStrategy(checkers_swig.Team_Black, net_path, True)
    env = checkers_swig.CheckersEnv(True)
    result = env.Run(random, qnn_2)
    if result.winner == 0:
        stats['white'] += 1
    elif result.winner == 1:
        stats['black'] += 1
    elif result.winner == 2:
        stats['draw'] += 1
print(stats)

print('play white sample')
stats = {'white': 0, 'black': 0, 'draw': 0}
for i in range(1000):
    qnn_1 = checkers_swig.MakeA2CStrategy(checkers_swig.Team_White, net_path, False)
    env = checkers_swig.CheckersEnv(True)
    result = env.Run(qnn_1, random)
    if result.winner == 0:
        stats['white'] += 1
    elif result.winner == 1:
        stats['black'] += 1
    elif result.winner == 2:
        stats['draw'] += 1
print(stats)

print('play black greedy')
stats = {'white': 0, 'black': 0, 'draw': 0}
for i in range(1000):
    qnn_2 = checkers_swig.MakeA2CStrategy(checkers_swig.Team_Black, net_path, False)
    env = checkers_swig.CheckersEnv(True)
    result = env.Run(random, qnn_2)
    if result.winner == 0:
        stats['white'] += 1
    elif result.winner == 1:
        stats['black'] += 1
    elif result.winner == 2:
        stats['draw'] += 1
print(stats)
