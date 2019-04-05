import checkers_swig                                                                             

stream = checkers_swig.stringstream()                                                            
random_1 = checkers_swig.MakeRandomStrategy(666)                                                    
random_2 = checkers_swig.MakeRandomStrategy(999)
env = checkers_swig.CheckersEnv()
result = env.Run(random_1, random_2)
for state in result.story: 
    state.Dump(stream)
print(stream.str())
