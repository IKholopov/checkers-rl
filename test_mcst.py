import checkers_swig                                                                             

stream = checkers_swig.stringstream()                                                            
random_1 = checkers_swig.MakeRandomStrategy()                                                    
mcst_2 = checkers_swig.MakeMCSTStrategy(checkers_swig.Team_Black, 100)
env = checkers_swig.CheckersEnv()
result = env.Run(random_1, mcst_2)
for state in result.story: 
    state.Dump(stream)
print(stream.str())
