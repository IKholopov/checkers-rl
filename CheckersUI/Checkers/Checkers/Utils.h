// Автор: Николай Фролов.

#pragma once

#include <Board.h>
#include <GameState.h>

std::shared_ptr<GameState> BoardToGameState( const CBoard&, bool american, Team team );
CBoard GameStateToBoard( const GameState& );