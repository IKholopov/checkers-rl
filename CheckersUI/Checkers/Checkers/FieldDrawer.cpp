// Автор: Николай Фролов.

#include <FieldDrawer.h>

#include <cassert>

CFieldDrawer::CFieldDrawer()
{
	whiteBrush = ::CreateSolidBrush( RGB( 204, 204, 204 ) );
	kingWhiteBrush = ::CreateSolidBrush( RGB( 255, 255, 255 ) );
	blackBrush = ::CreateSolidBrush( RGB( 102, 102, 102 ) );
	kingBlackBrush = ::CreateSolidBrush( RGB( 51, 51, 51 ) );
	
	backgroundBrush = ::CreateSolidBrush( RGB( 184, 115, 51 ) ) ;
	backgroundPen = ::CreatePen( PS_SOLID, 10, RGB( 184, 115, 51 ) );

	focusedPen = ::CreatePen( PS_SOLID, 5, RGB( 51, 255, 0 ) );
	availablePen = ::CreatePen( PS_SOLID, 5, RGB( 255, 255, 0 ) );
}

CFieldDrawer::~CFieldDrawer()
{
	::DeleteObject( whiteBrush );
	::DeleteObject( kingWhiteBrush );
	::DeleteObject( blackBrush );
	::DeleteObject( kingBlackBrush );

	::DeleteObject( backgroundBrush );
	::DeleteObject( backgroundPen );

	::DeleteObject( focusedPen );
	::DeleteObject( availablePen );
}

void CFieldDrawer::DrawField( const CField& field ) const
{
	PAINTSTRUCT paintInfo;
	HDC displayHandle = ::BeginPaint( field.Window, &paintInfo );
	assert( displayHandle != 0 );

	RECT rectInfo = paintInfo.rcPaint;
	int width = rectInfo.right;
	int height = rectInfo.bottom;

	HDC tempHDC = ::CreateCompatibleDC( displayHandle );
	HBITMAP tempBitmap = ::CreateCompatibleBitmap( displayHandle, width, height );
	HBITMAP oldBitmap = static_cast<HBITMAP>( ::SelectObject( tempHDC, tempBitmap ) );

	drawBackground( field, tempHDC, rectInfo );
	drawChecker( field, tempHDC, rectInfo );
	
	::BitBlt( displayHandle, 0, 0, width, height, tempHDC, 0, 0, SRCCOPY );

	::DeleteObject( oldBitmap );
	::DeleteObject( tempBitmap );
	::DeleteObject( tempHDC );
	::EndPaint( field.Window, &paintInfo );
}

// Отрисовка фона игрового поля.
void CFieldDrawer::drawBackground( const CField& field, HDC tempHDC, RECT rectInfo ) const
{
	int width = rectInfo.right;
	int height = rectInfo.bottom;

	HBRUSH oldBrush = static_cast<HBRUSH>( ::SelectObject( tempHDC, backgroundBrush ) );
	HPEN oldPen;
	if( field.HasBorder ) {
		// Если у поля есть обводка, то ее цвет зависит от того, нажимал на нее пользователь или нет.
		if( field.Window == ::GetFocus() ) {
			oldPen = static_cast<HPEN>( ::SelectObject( tempHDC, focusedPen ) );
		} else {
			oldPen = static_cast<HPEN>( ::SelectObject( tempHDC, availablePen ) );
		}
	} else {
		// Если обводки нет, то цвет рамки тот же, что и цвет фона клетки.
		oldPen = static_cast<HPEN>( ::SelectObject( tempHDC, backgroundPen ) );
	}
	Rectangle( tempHDC, 0, 0, width, height );

	::SelectObject( tempHDC, oldPen );
	::SelectObject( tempHDC, oldBrush );
}

// Отрисовка шашки в игровом поле.
void CFieldDrawer::drawChecker( const CField& field, HDC tempHDC, RECT rectInfo ) const
{
	if( field.Condition != FC_Empty ) {
		HBRUSH oldBrush;
		int width = rectInfo.right;
		int height = rectInfo.bottom;

		// Определяем цвет шашки.
		if( field.Condition == FC_WhiteChecker ) {
			oldBrush = static_cast<HBRUSH>( ::SelectObject( tempHDC, whiteBrush ) );
		} else {
			oldBrush = static_cast<HBRUSH>( ::SelectObject( tempHDC, blackBrush ) );
		}
		::Ellipse( tempHDC, baseIndent, baseIndent, width - baseIndent, height - baseIndent );

		// Если это дамка, то делаем дополнительную отрисовку в зависимости от цвета шашки.
		if( field.IsKing ) {
			if( field.Condition == FC_WhiteChecker ) {
				::SelectObject( tempHDC, kingWhiteBrush );
			} else {
				::SelectObject( tempHDC, kingBlackBrush );
			}
			::Ellipse( tempHDC, baseIndent + kingIndent, baseIndent + kingIndent, width - baseIndent - kingIndent, height - baseIndent - kingIndent );
		}

		::SelectObject( tempHDC, oldBrush );
	}
}