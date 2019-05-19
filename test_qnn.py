import checkers_swig                                                                             

stream = checkers_swig.stringstream()
print('play white')
random = checkers_swig.MakeRandomStrategy(100)
stats = {'white': 0, 'black': 0, 'draw': 0}
for i in range(1000):
    qnn_1 = checkers_swig.MakeQNNStrategy(checkers_swig.Team_White, 'CheckpointMinMax.trpt')
    env = checkers_swig.CheckersEnv(True)
    result = env.Run(qnn_1, random)
    if result.winner == 0:
        stats['white'] += 1
    elif result.winner == 1:
        stats['black'] += 1
    elif result.winner == 2:
        stats['draw'] += 1
print(stats)

print('play black')
stats = {'white': 0, 'black': 0, 'draw': 0}
for i in range(1000):
    qnn_2 = checkers_swig.MakeQNNStrategy(checkers_swig.Team_Black, 'CheckpointMinMax.trpt')
    env = checkers_swig.CheckersEnv(True)
    result = env.Run(random, qnn_2)
    if result.winner == 0:
        stats['white'] += 1
    elif result.winner == 1:
        stats['black'] += 1
    elif result.winner == 2:
        stats['draw'] += 1
print(stats)
