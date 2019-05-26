// Автор: Фролов Николай.

#include <CheckersEngine.h>
#include <Board.h>

#include <Strategy.h>
#include <StrategyFactory.h>
#include <Utils.h>

#include <algorithm>
#include <cassert>

CCheckersEngine::CCheckersEngine( CBoard& _board )
	: board( _board )
	, playBoard( board.GetBoard() )
	, isWhiteTurn( true )
	, result( GR_StillPlaying )
{
	opponentStrategy = Strategy::MakeQNNStrategy( Team::Black, "C:\\Users\\Nikolay\\checkers-rl\\CheckpointMinMax.trpt" );
}

void CCheckersEngine::StartGame()
{
	// Выставляем все параметры в стартовое состояние.
	board.Reset();
	possibleTurns.clear();
	clearDrawCheck();
	isWhiteTurn = false;
	isTurnHasTakings = false;
	::SetFocus( mainWindowHandle );
	result = GR_StillPlaying;

	for( auto& field : playBoard ) {
		::InvalidateRect( field.Window, 0, true );
	}

	// Начинаем игру, запуская расчет доступных первых ходов.
	calculateNextTurn();
}

void CCheckersEngine::AddFocus( int fieldIdx )
{
	// Получаем доступные из данного поля ходы.
	std::list< std::deque<int> >& possibleTurnsForField = possibleTurns[fieldIdx];
	// Подсвечиваем и перерисовываем поля, в которые можно совершить ходы из fieldIdx.
	for( auto& possibleTurn : possibleTurnsForField ) {
		playBoard[possibleTurn[isTurnHasTakings ? 1 : 0]].HasBorder = true;
		::InvalidateRect( playBoard[possibleTurn[isTurnHasTakings ? 1 : 0]].Window, 0, true );
	}
	// Перерисовываем само окно с fieldIdx полем - теперь оно выделено другим цветом.
	::InvalidateRect( playBoard[fieldIdx].Window, 0, true );
}

void CCheckersEngine::DelFocus( int fieldIdx )
{
	// Если из данной клетки возможных ходов уже нет, то выделение уже убрано.
	if( possibleTurns.find( fieldIdx ) == possibleTurns.end() ) {
		return;
	}
	// Получаем доступные из данного поля ходы.
	std::list< std::deque<int> >& possibleTurnsForField = possibleTurns[fieldIdx];
	// Убираем подсветку полей, в которые можно совершить ходы из fieldIdx и перерисовываем их.
	for( auto& possibleTurn : possibleTurnsForField ) {
		playBoard[possibleTurn[isTurnHasTakings ? 1 : 0]].HasBorder = false;
		::InvalidateRect( playBoard[possibleTurn[isTurnHasTakings ? 1 : 0]].Window, 0, true );
	}
	// Перерисовываем само окно с fieldIdx полем - теперь у него нет специального выделения.
	::InvalidateRect( playBoard[fieldIdx].Window, 0, true );
}

void CCheckersEngine::TryTurn( int from, int to )
{
	std::list< std::deque<int> >& possibleTurnsForField = possibleTurns[from];
	// Флаг - получилось ли сделать ход из поля from в поле to.
	bool hasPossibleTurn = false;
	// Здесь окажутся остатки доступных ходов, в том случае, если доступно множественное взятие.
	std::list< std::deque<int> > restOfTurns;
	for( auto& possibleTurn : possibleTurnsForField ) {
		// В зависимости от того, есть в данном ходу взятие или нет, меняется формат хранения ходов.
		if( possibleTurn[isTurnHasTakings ? 1 : 0] == to ) {
			// Если взятия есть и нашелся подходящий ход, то нужно снять изъять из описания хода две ячейки: одна их них
			// содержит номер поля, на котором стоит срубаемая шашка, на другом номер, в которой происходит перемещение.
			if( isTurnHasTakings ) {
				playBoard[*possibleTurn.begin()].Condition = FC_Empty;
				playBoard[*possibleTurn.begin()].IsKing = false;
				::InvalidateRect( playBoard[*possibleTurn.begin()].Window, 0, true );
				possibleTurn.pop_front();
			}
			possibleTurn.pop_front();
			if( !possibleTurn.empty() ) {
				restOfTurns.push_back( possibleTurn );
			}
			hasPossibleTurn = true;
		}
	}

	if( hasPossibleTurn ) {
		makePossibleTurn( from, to );
		handleRestOfTurns( to, restOfTurns );
	}
}

void CCheckersEngine::MakeAITurn()
{
	isWhiteTurn = true;
	isTurnHasTakings = false;
	if( calculateNextTurn() ) {

		std::shared_ptr<GameState> currentState = BoardToGameState( board, true, Team::White );
		std::shared_ptr<GameState> nextState = opponentStrategy->ChooseNextState( currentState );

		CBoard newBoardState = GameStateToBoard( *nextState );
		std::vector<CField>& newFields = newBoardState.GetBoard();

		for( int i = 0; i < playBoard.size(); i++ ) {
			playBoard[i].Condition = newFields[i].Condition;
			playBoard[i].IsKing = newFields[i].IsKing;
			playBoard[i].HasBorder = false;
			::InvalidateRect( playBoard[i].Window, 0, true );
		}
	
		possibleTurns.clear();
		isTurnHasTakings = false;
		isWhiteTurn = false;
		calculateNextTurn();
	}
}

// Рассчитать возможные следующие ходы.
bool CCheckersEngine::calculateNextTurn()
{
	// Ходы рассчитываем перебором. Для более быстрого перебора формируем доску в сокращенном виде
	// Здесь каждое поле описывается парой вида <Состояние поля; является ли пешка на поле дамкой>.
	shortcutPlayBoard.resize( playBoard.size() );
	for( size_t i = 0; i < playBoard.size(); ++i ) {
		shortcutPlayBoard[i] = std::pair<TFieldCondition, bool>( playBoard[i].Condition, playBoard[i].IsKing );
	}

	if( isWhiteTurn ) {
		ally = FC_WhiteChecker;
		enemy = FC_BlackChecker;
	} else {
		ally = FC_BlackChecker;
		enemy = FC_WhiteChecker;
	}

	for( size_t i = 0; i < playBoard.size(); ++i ) {
		if( playBoard[i].Condition == ally ) {
			calculatePossibleTurnsForField( i );
		}
	}

	// У текущего игрока нет ни одного допустимого хода(все его шашки взяты/заблокированы), значит он проиграл.
	// Либо до этого была диагностирована ничья, и ходы все еще остались - то есть победитель не был определен прошлым ходом.
	if( possibleTurns.empty() || result == GR_Draw ) {
		if( result != GR_Draw ) {
			if( isWhiteTurn ) {
				result = GR_BlackWon;
			} else {
				result = GR_WhiteWon;
			}
		}
		endGame();
		return false;
	}

	// Добавляем подсветку к полям, из которых доступны ходы.
	for( auto& possibleTurn: possibleTurns ) {
		if( possibleTurn.first >= 0 ) {
			playBoard[possibleTurn.first].HasBorder = true;
		}
		// Если возможны взятия, то нужно убрать из описаний ходов, некоторую доп. информацию, которая более не нужна.
		if( isTurnHasTakings ) {
			for( auto& turnWithExtraInfo : possibleTurn.second ) {
				turnWithExtraInfo.pop_front();
			}
		}
		::InvalidateRect( playBoard[possibleTurn.first].Window, 0, true );
	}
	return true;
}

// Расчитываем возможный ход из поля fieldIdx.
// В calculatedTurn находится последовательность уже сделанных в данном ходу переходов.
void CCheckersEngine::calculatePossibleTurnsForField( int fieldIdx )
{
	// Расчитываем доступные ходы, в зависимости от того, является ли шашка дамкой или нет.
	if( !shortcutPlayBoard[fieldIdx].second ) {
		calculateNonKingTurn( fieldIdx, std::deque<int>() );
	} else {
		calculateKingTurn( fieldIdx, std::deque<int>() );
	}
}

// Расчет хода обычной шашки, находящейся на поле fieldIdx, calculatedTurn - уже рассчитанная часть хода.
void CCheckersEngine::calculateNonKingTurn( int fieldIdx, std::deque<int>& calculatedTurn )
{
	// Получаем список вершин с соседних полудиагоналей. В них нас для обычной шашки интересуют
	// только ближайшие к данном клетке два элемента, т.к. для хода доступны только они.
	const std::vector< std::vector<int> >& neighbours = calculateNeighbourFields( fieldIdx );
	// Флаг, который нужен для того, чтобы при взятии в результате рекурсии ход попал в список доступных.
	bool IsTriedToAddTurn = false;
	for( size_t i = 0; i < neighbours.size(); ++i ) {
		if( ally == FC_WhiteChecker && fieldIdx <= neighbours[i][0]
			|| ally == FC_BlackChecker && fieldIdx >= neighbours[i][0] )
		{
			continue;
		}
		// Возможны два случая - либо шашка производит взятие, либо нет - соответственно обрабатываем каждый из них.
		if( shortcutPlayBoard[neighbours[i][0]].first == FC_Empty && calculatedTurn.empty() 
			&& ( ( neighbours[i][0] < fieldIdx && isWhiteTurn ) || ( neighbours[i][0] > fieldIdx && !isWhiteTurn ) ) ) {
			calculatedTurn.push_back( neighbours[i][0] );
			tryAddTurn( fieldIdx, calculatedTurn );
			IsTriedToAddTurn = true;
			calculatedTurn.pop_back();
		} else if( shortcutPlayBoard[neighbours[i][0]].first == enemy && neighbours[i].size() > 1 && shortcutPlayBoard[neighbours[i][1]].first == FC_Empty ) {
			IsTriedToAddTurn = true;
			if( calculatedTurn.empty() ) {
				calculatedTurn.push_back( fieldIdx );
			}
			calculatedTurn.push_back( neighbours[i][0] );
			calculatedTurn.push_back( neighbours[i][1] );
			shortcutPlayBoard[neighbours[i][0]].first = FC_Empty;
			std::swap( shortcutPlayBoard[fieldIdx], shortcutPlayBoard[neighbours[i][1]] );
			calculateNonKingTurn( neighbours[i][1], calculatedTurn );
			std::swap( shortcutPlayBoard[fieldIdx], shortcutPlayBoard[neighbours[i][1]] );
			shortcutPlayBoard[neighbours[i][0]].first = enemy;
			calculatedTurn.pop_back();
			calculatedTurn.pop_back();
		}
	}
	// Если это произошло, то в calculatedTurn содержится последовательность взятий, которую нельзя продолжить.
	if( !IsTriedToAddTurn && !calculatedTurn.empty() ) {
		tryAddTurn( calculatedTurn[0], calculatedTurn );
	}
}

// Расчет хода дамки, находящейся на поле fieldIdx, calculatedTurn - уже рассчитанная часть хода.
void CCheckersEngine::calculateKingTurn( int fieldIdx, std::deque<int>& calculatedTurn )
{	
	// Получаем список вершин с соседних полудиагоналей.
	const std::vector< std::vector<int> >& neighbours = calculateNeighbourFields( fieldIdx );
	// Флаг, который нужен для того, чтобы при взятии в результате рекурсии ход попал в список доступных.
	bool IsTriedToAddTurn = false;
	for( size_t i = 0; i < neighbours.size(); ++i ) {
		int metEnemyIdx = -1;
		for( size_t j = 0; j < min( 2, neighbours[i].size() ); ++j ) {
			// Если наткнулись на шашку того же цвета, что и ходящая, либо на два противоположного, стоящие в ряд
			// то дальше расчитывать ход смысла нет.
			if( shortcutPlayBoard[neighbours[i][j]].first == ally 
				|| ( shortcutPlayBoard[neighbours[i][j]].first == enemy && metEnemyIdx != -1 ) ) {
				break;
			// Если впервые на линии наткнулись на шашку противположного цвета, то запоминаем ее позицию.
			} else if( shortcutPlayBoard[neighbours[i][j]].first == enemy ) {
				metEnemyIdx = neighbours[i][j];
			// Если наткнулись на пустое поле и до этого не встречали шашек противника, то эта клетка доступна для хода.
			} else if( calculatedTurn.empty() && shortcutPlayBoard[neighbours[i][j]].first == FC_Empty && metEnemyIdx == -1 ) {
				calculatedTurn.push_back( neighbours[i][j] );
				tryAddTurn( fieldIdx, calculatedTurn );
				IsTriedToAddTurn = true;
				calculatedTurn.pop_back();
			// Если наткнулись на пустое поле и встретив до этого шашку противника, можем произвести взятие.
			} else if( shortcutPlayBoard[neighbours[i][j]].first == FC_Empty && metEnemyIdx != -1 ) {
				IsTriedToAddTurn = true;
				if( calculatedTurn.empty() ) {
					calculatedTurn.push_back( fieldIdx );
				}
				calculatedTurn.push_back( metEnemyIdx );
				calculatedTurn.push_back( neighbours[i][j] );
				shortcutPlayBoard[metEnemyIdx].first = FC_Empty;
				std::swap( shortcutPlayBoard[fieldIdx], shortcutPlayBoard[neighbours[i][j]] );
				calculateKingTurn( neighbours[i][j], calculatedTurn );
				std::swap( shortcutPlayBoard[fieldIdx], shortcutPlayBoard[neighbours[i][j]] );
				shortcutPlayBoard[metEnemyIdx].first = enemy;
				calculatedTurn.pop_back();
				calculatedTurn.pop_back();
			}
		}
	}
	// Если это произошло, то в calculatedTurn содержится последовательность взятий, которую нельзя продолжить.
	if( !IsTriedToAddTurn && !calculatedTurn.empty() ) {
		tryAddTurn( calculatedTurn[0], calculatedTurn );
	}
}

// Получить элемент отображения calculatedNeighbourFields, связанный с клеткой fieldIdx.
const std::vector< std::vector<int> >& CCheckersEngine::calculateNeighbourFields( int fieldIdx ) const
{
	if( calculatedNeighbourFields.find( fieldIdx ) == calculatedNeighbourFields.end() ) {
		int numberOfCheckersInRow = board.BoardSize / 2;
		// Для удобства вычислений переводим номер клетки, пару номеров, которая бы соответствовала
		// клетке в квадратной матрице размером BoardSize * BoardSize.
		int i = fieldIdx / numberOfCheckersInRow;
		int j = ( fieldIdx % numberOfCheckersInRow ) * 2 + ( ( fieldIdx / numberOfCheckersInRow + 1 ) % 2 );
		std::vector<int> currentDiag;

		// Пытаемся поочередной добавить в порядке удаления соседей о всех четырех полудиагоналей.
		for( int k = 1; k < std::min<int>( i + 1, j + 1 ); ++k ) {
			currentDiag.push_back( ( i - k ) * numberOfCheckersInRow + ( j - k ) / 2 );
		}
		if( !currentDiag.empty() ) {
			calculatedNeighbourFields[fieldIdx].push_back( currentDiag );
			currentDiag.clear();
		}

		for( int k = 1; k < std::min<int>( i + 1, board.BoardSize - j ); ++k ) {
			currentDiag.push_back( ( i - k ) * numberOfCheckersInRow + ( j + k ) / 2 );
		}
		if( !currentDiag.empty() ) {
			calculatedNeighbourFields[fieldIdx].push_back( currentDiag );
			currentDiag.clear();
		}

		for( int k = 1; k < std::min<int>( board.BoardSize - i, j + 1 ); ++k ) {
			currentDiag.push_back( ( i + k ) * numberOfCheckersInRow + ( j - k ) / 2 );
		}
		if( !currentDiag.empty() ) {
			calculatedNeighbourFields[fieldIdx].push_back( currentDiag );
			currentDiag.clear();
		}

		for( int k = 1; k < std::min<int>( board.BoardSize - i, board.BoardSize - j ); ++k ) {
			currentDiag.push_back( ( i + k ) * numberOfCheckersInRow + ( j + k ) / 2 );
		}
		if( !currentDiag.empty() ) {
			calculatedNeighbourFields[fieldIdx].push_back( currentDiag );
		}
	}
	return calculatedNeighbourFields[fieldIdx];
}

// Попытка добавить к возможным ходам ход, описываемыый последовательностью calculatedTurn.
// Ход невозможно добавить, если уже есть ходы, в которых происходит больше взятий, чем в описанном.
void CCheckersEngine::tryAddTurn( int fromField, std::deque<int>& calculatedTurn )
{
	// Если размер последовательности больше 1, то есть взятия .
	if( calculatedTurn.size() > 1 ) {
		isTurnHasTakings = true;
	}
	// Проверяем, что в возможные ходы в соответствие с правилами попадут лишь ходы с наибольшим числом взятий.
	if( possibleTurns.empty() || possibleTurns.begin()->second.begin()->size() == calculatedTurn.size() ) {
		possibleTurns[fromField].push_back( calculatedTurn );
	} else if( possibleTurns.begin()->second.begin()->size() < calculatedTurn.size() ) {
		possibleTurns.clear();
		possibleTurns[fromField].push_back( calculatedTurn );
	}
}

// Выполняем ход из from в to, для которого уже определена доступность.
void CCheckersEngine::makePossibleTurn( int from, int to )
{
	// Совершаем перемещение шашки из from в to, убираем доступные ходы и выделение доступных клеток.
	playBoard[to].Condition = playBoard[from].Condition;
	playBoard[from].Condition = FC_Empty;
	playBoard[to].IsKing = playBoard[from].IsKing;
	playBoard[from].IsKing = false;
	possibleTurns.clear();
	for( auto& field : playBoard ) {
		if( field.HasBorder ) {
			field.HasBorder = false;
			::InvalidateRect( field.Window, 0, true );
		}
	}
}

// Завершаем ход или обрабатываем его остаток, в зависимости от содержания массива restOfTurns.
void CCheckersEngine::handleRestOfTurns( int newTurnPosition, std::list< std::deque<int> >& restOfTurns )
{
	// Если ход еще не закончен, то отмечаем новые доступные клетки и обновляем доступные ходы.
	if( !restOfTurns.empty() ) {
		for( auto& turn : restOfTurns ) {
			playBoard[turn[1]].HasBorder = true;
			::InvalidateRect( playBoard[turn[1]].Window, 0, true );
		}
		playBoard[newTurnPosition].HasBorder = true;
		possibleTurns[newTurnPosition] = restOfTurns;
		::InvalidateRect( playBoard[newTurnPosition].Window, 0, true );
	} else {
		// Если ход закончен, то проверяем, не превратилась ли шашка в дамку.
		if( !playBoard[newTurnPosition].IsKing ) {
			if( ( newTurnPosition < board.BoardSize / 2 && isWhiteTurn ) 
				|| ( newTurnPosition >= ( static_cast<int>( playBoard.size() ) - board.BoardSize / 2 ) && !isWhiteTurn ) ) {
				playBoard[newTurnPosition].IsKing = true;
				::InvalidateRect( playBoard[newTurnPosition].Window, 0, true );
			}
		}
		// Ход завершен. Проверяем выполнения одного из условий ничьи.
		checkDraw( newTurnPosition );
		// Если ход окончен, то передаем ход другой стороне и рассчитываем ее доступные ходы.
		MakeAITurn();
	}
}

// Проверяет выполнение условий ничьи.
void CCheckersEngine::checkDraw( int finishedTurnField  )
{
	if( checkDrawCondition1( finishedTurnField  ) || checkDrawCondition2()  ) {
		result = GR_Draw;
	}
}

// Игроки в течение 25 ходов делали ходы только дамками, не передвигая простых шашек и не производя взятия.
bool CCheckersEngine::checkDrawCondition1( int finishedTurnField  )
{
	if( !isTurnHasTakings && playBoard[finishedTurnField].IsKing ) {
		++numberOfTurnsWithOnlyKings;
	} else {
		numberOfTurnsWithOnlyKings = 0;
	}
	if( numberOfTurnsWithOnlyKings >= 25 ) {
		drawReason = DR_Condition1;
		return true;
	}
	return false;
}

// Три раза повторяется одна и та же позиция, причём очередь хода каждый раз будет за одной и той же стороной.
bool CCheckersEngine::checkDrawCondition2()
{
	// Сокращенное описание доски.
	std::string boardDescription;
	// Первый символ описывает того, был последний ход. 0 - белых, 1 - черных.
	boardDescription.push_back( isWhiteTurn ? '0' : '1' );
	// Далее 50 символов: 0 - пустое поле, 1 - белая шашка, 2 - белая дамка, 3 - черная шашка, 4 - черная дамка.
	for( auto& field : playBoard ) {
		switch( field.Condition ) {
			case FC_Empty:
				boardDescription.push_back( '0' );
				break;
			case FC_WhiteChecker:
				if( field.IsKing ) {
					boardDescription.push_back( '2' );
				} else {
					boardDescription.push_back( '1' );
				}
				break;
			case FC_BlackChecker:
				if( field.IsKing ) {
					boardDescription.push_back( '4' );
				} else {
					boardDescription.push_back( '3' );
				}
				break;
			default:
				assert( false );
		}
	}
	if( ++finishedTurns[boardDescription] >= 3 ) {
		drawReason = DR_Condition2;
		return true;
	}
	return false;
}

// Очистка данных о ничьи. Выполняется при запуске новой игры.
void CCheckersEngine::clearDrawCheck()
{
	numberOfTurnsWithOnlyKings = 0;
	finishedTurns.clear();
}

// Конец партии. Сообщаем игрокам о результате.
void CCheckersEngine::endGame()
{
	wchar_t* message = 0;
	wchar_t* caption = 0;
	if( result != GR_Draw ) {
		caption = L"Victory!";
		if( result == GR_BlackWon ) {
			message = L"Congratulations! Blacks won!\nDo you want to start a new game?";
		} else {
			message = L"Congratulations! Whites won!\nDo you want to start a new game?";
		}
	} else {
		caption = L"Draw.";
		if( drawReason == DR_Condition1 ) {
			message = L"Draw. Reason: 25 turns with only kings without captures.\nDo you want to start a new game?";
		} else {
			message = L"Draw. Reason: Same positions occured 3 times.\nDo you want to start a new game?";
		}
	}
	int answer;
	answer = ::MessageBox( mainWindowHandle, message, caption, MB_YESNO );
	if( answer == IDYES ) {
		StartGame();
	}
}