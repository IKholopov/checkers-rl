// Автор: Фролов Николай.

#include <MainWindow.h>
#include <resource.h>

CMainWindow::CMainWindow()
	: board( boardSize )
	, fields()
	, focusedWindowIdx( -1 )
	, engine( board )
{
	for( int  i = 0; i < board.BoardSize * board.BoardSize / 2; ++i ) {
		fields.push_back( CFieldWindow( board.GetBoard()[i], focusedWindowIdx, engine ) );
	}
}

bool CMainWindow::RegisterClass()
{
    WNDCLASSEX windowWND;

    windowWND.cbSize = sizeof( WNDCLASSEX );
    windowWND.style = CS_HREDRAW | CS_VREDRAW;
    windowWND.lpfnWndProc = mainWindowProc;
    windowWND.cbClsExtra = 0;
    windowWND.cbWndExtra = 0;
    windowWND.hInstance = static_cast<HINSTANCE>( GetModuleHandle( 0 ) );
    windowWND.hIcon = 0;
    windowWND.hCursor = ::LoadCursor( 0, IDC_ARROW );
    windowWND.hbrBackground = ::CreateSolidBrush( RGB( 207, 236, 255 ) );
    windowWND.lpszMenuName = MAKEINTRESOURCE( IDR_MENU1 );
    windowWND.lpszClassName = L"CMainWindow";
    windowWND.hIconSm = 0;

    return ::RegisterClassEx( &windowWND ) != 0;
}

bool CMainWindow::Create()
{
	int realWidth = width + ::GetSystemMetrics( SM_CXSIZEFRAME ) * 2 + ::GetSystemMetrics( SM_CXPADDEDBORDER ) * 2;
	int realHeight = height + ::GetSystemMetrics( SM_CYCAPTION )+ ::GetSystemMetrics( SM_CYSIZEFRAME ) * 2
		+ ::GetSystemMetrics( SM_CXPADDEDBORDER ) * 2 + ::GetSystemMetrics( SM_CXMENUSIZE );
    handle = ::CreateWindowEx( 0, L"CMainWindow", L"Checkers", WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
		0, 0, realWidth, realHeight, 0, 0, static_cast<HINSTANCE>( ::GetModuleHandle( 0 ) ), this );
	engine.SetMainWindowHandle( handle );

    return handle != 0;
}

void CMainWindow::Show( int cmdShow ) const
{
    ::ShowWindow( handle, cmdShow );
	for( size_t i = 0; i < fields.size(); ++i ) {
		fields[i].Show( cmdShow );
	}
}

void CMainWindow::OnDestroy() const
{
    ::PostQuitMessage( 0 );
}

LRESULT CMainWindow::wmCommandProc( HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam )
{
	CMainWindow* window = reinterpret_cast<CMainWindow*>( ::GetWindowLongPtr( hwnd, GWLP_USERDATA ) );

	switch( HIWORD( wParam ) ) {
		case 0:
			switch( LOWORD( wParam ) ) {
				case ID_MENU_NEWGAME:
					window->engine.StartGame();
					break;
				case ID_MENU_EXIT:
					window->OnDestroy();
					break;
				default:
					return ::DefWindowProc( hwnd, message, wParam, lParam );
			}
		default:
			return ::DefWindowProc( hwnd, message, wParam, lParam );
	}
	return 0;
}

// Создание массива дочерных окон, каждое из которых отвечает за одну игровую клетку.
void CMainWindow::createChildren( HWND hwnd )
{
	size_t numberOfCheckersInOneLine = boardSize / 2;
	for( size_t i = 0; i < fields.size(); ++i ) {
		int xStart = ( ( i % numberOfCheckersInOneLine ) * 2 + ( ( i / numberOfCheckersInOneLine + 1 ) % 2 ) ) * fieldSize;
		int yStart = i / numberOfCheckersInOneLine * fieldSize;
		fields[i].Create( hwnd, xStart, yStart, fieldSize, fieldSize );
	}
}

LRESULT __stdcall CMainWindow::mainWindowProc( HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam )
{
	CMainWindow* window = reinterpret_cast<CMainWindow*>( ::GetWindowLongPtr( hwnd, GWLP_USERDATA ) );

    switch( message ) {
		case WM_NCCREATE:
			::SetWindowLongPtr( hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>( reinterpret_cast<CREATESTRUCT*>( lParam )->lpCreateParams ) );
			return ::DefWindowProc( hwnd, message, wParam, lParam );
			break;
		case WM_CREATE:
            window->createChildren( hwnd );
			window->engine.StartGame();
			return 1;
		case WM_DESTROY:
			window->OnDestroy();
			break;
		case WM_COMMAND:
			return wmCommandProc( hwnd, message, wParam, lParam);
		case WM_WINDOWPOSCHANGED:
			for( auto& ptr : window->board.GetBoard() ) {
				::InvalidateRect( ptr.Window, 0, true );
			}
			break;
		default:
			return ::DefWindowProc( hwnd, message, wParam, lParam );
    }
    return 0;
}