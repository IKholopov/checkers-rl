// Автор: Николай Фролов.

#include <Utils.h>

static int boardToGameStateIndex( int index, int boardSize )
{
	const int y = index * 2 / boardSize;
	const int x = ( y + 1 ) % 2 + index * 2 % boardSize;

	return x + y * boardSize;
}

static int gameStateToBoardIndex( int index, int boardSize )
{
	const int y = index / boardSize;
	const int x = index % boardSize;

	return ( x % 2 == y % 2 ) ? -1 : y * boardSize / 2 + x / 2;
}

std::shared_ptr<GameState> BoardToGameState( const CBoard& board, bool american, Team team )
{
	std::shared_ptr<GameState> gameState( new GameState( american, team ) );
	std::array<CellStatus, BoardCells>& state = gameState->State;
	for( int i = 0; i < state.size(); i++ ) {
		state[i] = CellStatus::None;
	}

	const std::vector<CField>& fields = board.GetBoard();
	for( int idx = 0; idx < fields.size(); idx++ ) {
		const CField& field = fields[idx];

		const int gameStateIdx = boardToGameStateIndex( idx, BoardSize );
		CellStatus& cellStatus = state[gameStateIdx];
		switch( field.Condition ) {
			case FC_Empty:
				cellStatus = CellStatus::None;
				break;
			case FC_BlackChecker:
				cellStatus = field.IsKing ? CellStatus::BlackQueen : CellStatus::Black;
				break;
			case FC_WhiteChecker:
				cellStatus = field.IsKing ? CellStatus::WhiteQueen : CellStatus::White;
				break;
			default:
				assert( false );
		}
	}

	return gameState;
}

CBoard GameStateToBoard( const GameState& gameState )
{
	CBoard board( BoardSize );
	std::vector<CField>& fields = board.GetBoard();
	for( int idx = 0; idx < gameState.State.size(); idx++ ) {
		const CellStatus& cellStatus = gameState.State[idx];
		const int boardIdx = gameStateToBoardIndex( idx, BoardSize );
		if( boardIdx == -1 ) {
			continue;
		}

		CField& field = fields[boardIdx];
		switch( cellStatus ) {
			case CellStatus::None:
				field.Condition = FC_Empty;
				field.IsKing = false;
				break;
			case CellStatus::Black:
				field.Condition = FC_BlackChecker;
				field.IsKing = false;
				break;
			case CellStatus::White:
				field.Condition = FC_WhiteChecker;
				field.IsKing = false;
				break;
			case CellStatus::BlackQueen:
				field.Condition = FC_BlackChecker;
				field.IsKing = true;
				break;
			case CellStatus::WhiteQueen:
				field.Condition = FC_WhiteChecker;
				field.IsKing = true;
				break;
			default:
				assert( false );
				break;
		}
	}

	return board;
}