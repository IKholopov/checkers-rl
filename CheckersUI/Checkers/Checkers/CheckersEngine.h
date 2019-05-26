// Автор: Фролов Николай.

// Описание: класс, отвечающий за логику игры в партии.

#pragma once

#include <Board.h>

#include <map>
#include <vector>
#include <deque>
#include <list>
#include <string>

struct IStrategy;

enum TGameResult {
	GR_StillPlaying,
	GR_WhiteWon,
	GR_BlackWon,
	GR_Draw
};

class CCheckersEngine {
public:
	CCheckersEngine( CBoard& _board );

	// Запустить начало игры.
	void StartGame();
	// Проверить доступен ли в данный момент ход из клетки from, в клетку to. Если да, то выполнить его.
	void TryTurn( int from, int to );
	// Отметить клетку под номеров fieldIdx как выбранную, и отобразить для нее возможные ходы.
	void AddFocus( int fieldIdx );
	// Отменить выделение для клетки fieldIdx и прекратить отображение возможных из нее ходов.
	void DelFocus( int fieldIdx );

	void MakeAITurn();
	// Установка описателя главного окна. Нужно для модальности бокса завершения игры.
	void SetMainWindowHandle( HWND handle ) { mainWindowHandle = handle; }
	
private:
	// Описывает какая из-за какой причины объявлена ничья.
	// Номера соответствуют функциям проверки 1-2 соответственно.
	enum TDrawReason {
		DR_Condition1,
		DR_Condition2,
	};
	std::shared_ptr<IStrategy> opponentStrategy;
	// Описание игровой доски.
	CBoard& board;
	std::vector<CField>& playBoard;
	// Описатель главного окна игры.
	HWND mainWindowHandle;
	// Определяет чей сейчас ход.
	bool isWhiteTurn;
	// В данную переменную попадает состояние игры.
	TGameResult result;
	// В данную переменную попадает причина ничьи.
	TDrawReason drawReason;

	// Здесь кешируется информация, связанные с вычислением следующих возможных ходов.
	std::map< int, std::list< std::deque<int> > > possibleTurns;
	std::vector< std::pair<TFieldCondition, bool> > shortcutPlayBoard;
	TFieldCondition ally;
	TFieldCondition enemy;

	// Отображение клеток в массивы клеток, которые находятся с данной на одной диагонали.
	// В каждом случае может быть до четырех таких массивов, в которых хранятся клетки
	// от данной до конца одной из диагоналей, не включая стартовую.
	mutable std::map< int, std::vector< std::vector<int> > > calculatedNeighbourFields;
	// Определяет есть ли доступные взятия в следующем ходу.
	mutable bool isTurnHasTakings;

	//Следующие параметры нужны для определения выполнения условий ничьи.
	// Счетчик для условия 25 ходов только дамками без взятий.
	int numberOfTurnsWithOnlyKings;
	// Отображение позиций на доске в количество их повторений. Текущее состояние доски кодируется строкой.
	std::map<std::string, int> finishedTurns;

	// Рассчитать возможные следующие ходы.
	bool calculateNextTurn();
	// Расчитываем возможный ход из поля fieldIdx.
	void calculatePossibleTurnsForField( int fieldIdx );
	// Расчет хода обычной шашки, находящейся на поле fieldIdx, calculatedTurn - уже рассчитанная часть хода.
	void calculateNonKingTurn( int fieldIdx, std::deque<int>& calculatedTurn );
	// Расчет хода дамки, находящейся на поле fieldIdx, calculatedTurn - уже рассчитанная часть хода.
	void calculateKingTurn( int fieldIdx, std::deque<int>& calculatedTurn );

	// Получить элемент отображения calculatedNonKingNeighbourFields, связанный с клеткой fieldIdx.
	// Если он еще не определен, то рассчитать его.
	const std::vector< std::vector<int> >& calculateNeighbourFields( int fieldIdx ) const;

	// Попытка добавить к возможным ходам ход, описываемый последовательностью calculatedTurn.
	// Ход невозможно добавить, если уже есть ходы, в которых происходит больше взятий, чем в описанном.
	void tryAddTurn( int fromField, std::deque<int>& calculatedTurn );
	// Выполняем ход из from в to, для которого уже определена доступность.
	void makePossibleTurn( int from, int to );
	// Завершаем ход или обрабатываем его остаток, в зависимости от содержания массива restOfTurns.
	void handleRestOfTurns( int newTurnPosition, std::list< std::deque<int> >& restOfTurns );

	// Проверяет выполнение условий ничьи.
	void checkDraw( int finishedTurnField );
	// Игроки в течение 25 ходов делали ходы только дамками, не передвигая простых шашек и не производя взятия.
	bool checkDrawCondition1( int finishedTurnField );
	// Три раза повторяется одна и та же позиция, причём очередь хода каждый раз будет за одной и той же стороной.
	bool checkDrawCondition2();
	// Очитска данных о ничьи. Выполняется при запуске новой игры.
	void clearDrawCheck();

	// Конец партии. Сообщаем игрокам о результате.
	void endGame();
};