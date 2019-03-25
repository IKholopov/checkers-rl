%module checkers_swig
%{
    #include <Env.h>
    #include <GameState.h>
    #include <StrategyFactory.h>
    #include <Strategy.h>
    #include <string>

    using namespace std;
%}

%include "std_sstream.i"
%include "std_iostream.i"
%include "std_string.i"

%include "std_shared_ptr.i"
%include <std_string.i>

%shared_ptr(GameState)
%shared_ptr(CheckersEnv)

%include "cpointer.i"
%include "std_vector.i"

// The order is important for SWIG. We probably should switch to Cython
// GameState does not have const qualifier, since I didn't manage to make it work with SWIG as well. Believe me, I tried.
%include "GameState.h"

namespace std {
    %template(vector_gamestate) vector<shared_ptr<GameState>>;
    %template(istrateg_ptr) shared_ptr<IStrategy>;
}
%include "Env.h"
%include "StrategyFactory.h"
