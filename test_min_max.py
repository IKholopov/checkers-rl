import checkers_swig                                                                             

stream = checkers_swig.stringstream()                                                            
random_1 = checkers_swig.MakeRandomStrategy(8)
minmax_2 = checkers_swig.MakeMinMaxStrategy(5)
env = checkers_swig.CheckersEnv()
result = env.Run(random_1, minmax_2)
for state in result.story: 
    state.Dump(stream)
print(stream.str())
