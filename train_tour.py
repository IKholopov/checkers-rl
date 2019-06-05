import torch
from tqdm import tqdm
import numpy as np
from a2c import A2CAgent, \
    train_against_strats, make_env, evaluate, save, \
    CheckersEnvWrapper, make_strat
import multiprocessing
import checkers_swig
from collections import defaultdict
from filelock import FileLock
import os


def do_train(strats, name, rootdir, steps, n_sync):
    os.chdir(rootdir)
    agent = A2CAgent((2, 8, 8))
    opt = torch.optim.Adam(agent.parameters(), lr=1e-3)
    dr = os.listdir(os.curdir)
    all_strats = strats + list(filter(lambda x: x[0] != '.', dr))
    lock = FileLock(os.path.join('.lock'))
    for st_info in train_against_strats(agent, opt, all_strats, 'rand', steps, n_sync):
        rewards_history, _, _, _, = st_info
        with lock:
            save(agent, name)
            all_strats = strats + os.listdir(os.curdir)
    os.chdir(os.pardir)

def evaluate_vs(evaluation, opponent, steps=50):
    stats = {'white': 0, 'black': 0, 'draw': 0}
    for _ in range(steps):
        qnn = checkers_swig.MakeA2CStrategy(checkers_swig.Team_White, evaluation, True)
        opp = make_strat(opponent)
        env = CheckersEnvWrapper(opp)
        result = env.simulate(qnn, opp())
        if result.winner == 0:
            stats['white'] += 1
        elif result.winner == 1:
            stats['black'] += 1
        elif result.winner == 2:
            stats['draw'] += 1
    return stats

def eval_models(models):
    scores = defaultdict(int)
    for i, model in enumerate(models):
        opponents = list(models[:i] + models[i+1:]) + ['minmax_5', 'mcst_rand', 'mcst_rand', 'rand']
        for op in tqdm(opponents):
            res = evaluate_vs(model, op)
            scores[model] += res['white'] - res['black'] - res['draw'] // 2
    return scores
