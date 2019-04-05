import checkers                                                                                  

env = checkers.CheckersAgent()                                                                   
state = env.observation()                                                                        
actions = env.possible_actions(state)                                                           
s, r, done, info = env.step(actions[0]) 
