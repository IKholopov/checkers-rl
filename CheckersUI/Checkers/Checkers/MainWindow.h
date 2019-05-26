// Автор: Фролов Николай.

// Описание: класс главного окна игры.

#pragma once

#include <FieldWindow.h>
#include <Board.h>
#include <CheckersEngine.h>

#include <Windows.h>
#include <vector>

class CMainWindow {
public:
	CMainWindow();

	// Регистрация класса окна.
    static bool RegisterClass();

    // Создание экземпляра окна.
    bool Create();

    // Показать окно.
    void Show( int cmdShow ) const;

protected:
    void OnDestroy() const;

private:
	// Описатель данного окна.
    HWND handle;

	// Размер одного поля.
	static const int fieldSize = 65;
	// Определяет размеры игровой доски.
	static const int boardSize = 8;
	// Количество шашек у каждого игрока в начале игры.
	static const int startNumberOfCheckers = 12;
	// Высота и ширина окна.
	static const int height = boardSize * fieldSize;
	static const int width = boardSize * fieldSize;
	
	// Игровое поле.
	CBoard board;
	// Поля, которые учавствуют в игре.
	std::vector<CFieldWindow> fields;
	// Класс, отвечающий за логику игры.
	CCheckersEngine engine;
	// Номер выбранного пользователем доступного для хода окна.
	// Если еще не выбрано ни одно таковое, то -1.
	int focusedWindowIdx;
	// Создание массива дочерных окон, каждое из которых отвечает за одну игровую клетку.
	void createChildren( HWND hwnd );

	static LRESULT __stdcall wmCommandProc( HWND hanlde, UINT message, WPARAM wParam, LPARAM lParam );
    static LRESULT __stdcall mainWindowProc( HWND hanlde, UINT message, WPARAM wParam, LPARAM lParam );
};