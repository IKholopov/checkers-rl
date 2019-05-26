// Автор: Фролов Николай.

// Описание: класс, отвечащий за отрисовку полей доски.

#pragma once

#include <Field.h>

#include <Windows.h>

class CFieldDrawer {
public:
	CFieldDrawer();
	~CFieldDrawer();

	// Отрисовка в окне window поля доски в соответствии с содержанием field.
	void DrawField( const CField& field ) const;

private:
	// Для отрисовки фигур.
	HBRUSH whiteBrush;
	HBRUSH kingWhiteBrush;
	HBRUSH blackBrush;
	HBRUSH kingBlackBrush;

	// Фон черных клеток.
	HBRUSH backgroundBrush;
	HPEN backgroundPen;

	// Для прорисовки подсказок.
	HPEN focusedPen;
	HPEN availablePen;

	// Отступ, для отрисовки шашек.
	static const int baseIndent = 3;
	// Отступ, при отрисовке дамок.
	static const int kingIndent = 5;

	// Отрисовка фона игрового поля.
	void drawBackground( const CField& field, HDC tempHDC, RECT rectInfo ) const;
	// Отрисовка шашки в игровом поле.
	void drawChecker( const CField& field, HDC tempHDC, RECT rectInfo ) const;
};