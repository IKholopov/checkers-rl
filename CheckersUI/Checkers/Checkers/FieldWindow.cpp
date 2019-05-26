// Автор: Фролов Николай.

#include <FieldWindow.h>
#include <FieldDrawer.h>

const CFieldDrawer CFieldWindow::drawer = CFieldDrawer();

CFieldWindow::CFieldWindow( CField& field, int& _focusedWindowIdx, CCheckersEngine& _engine )
	: windowField( field )
	, focusedWindowIdx( _focusedWindowIdx )
	, engine( _engine )
{
}

bool CFieldWindow::RegisterClass()
{
	WNDCLASSEX windowWND;

	windowWND.cbSize = sizeof( WNDCLASSEX );
	windowWND.style = CS_HREDRAW | CS_VREDRAW;
	windowWND.lpfnWndProc = fieldWindowProc;
	windowWND.cbClsExtra = 0;
	windowWND.cbWndExtra = 0;
	windowWND.hInstance = static_cast<HINSTANCE>( GetModuleHandle( 0 ) );
	windowWND.hIcon = 0;
	windowWND.hCursor = ::LoadCursor( 0, IDC_ARROW );
	windowWND.hbrBackground = 0;
	windowWND.lpszMenuName = 0;
	windowWND.lpszClassName = L"CFieldWindow";
	windowWND.hIconSm = 0;

	return ::RegisterClassEx( &windowWND ) != 0;
}

bool CFieldWindow::Create( HWND parent, int x, int y, int cx, int cy )
{
	handle = ::CreateWindowEx( 0, L"CFieldWindow", L"CFieldWindow", WS_CHILD, x, y, cx, cy, parent, 0,
		static_cast<HINSTANCE>( ::GetModuleHandle( 0 ) ), this );

	windowField.Window = handle;

	return handle != 0;
}

void CFieldWindow::Show( int cmdShow ) const
{
	::ShowWindow( handle, cmdShow );
}

void CFieldWindow::OnDestroy() const 
{
	::PostQuitMessage( 0 );
}

void CFieldWindow::OnPaint() const
{
	drawer.DrawField( windowField );
}

void CFieldWindow::OnLButtonDown() const
{
	if( windowField.HasBorder ) {
		// Если до этого была выделена клетка, то возмозно пытаются совершить ход.
		if( focusedWindowIdx != -1 && focusedWindowIdx != windowField.Name ) {
			engine.TryTurn( focusedWindowIdx, windowField.Name );
		}
		// Если после попытки хода, поле осталось выделено, значит либо ход совершить не удалось
		// либо ход соврешить удалось, но его можно продолжить.
		if( windowField.HasBorder ) {
			::SetFocus( handle );
			focusedWindowIdx = windowField.Name;
			engine.AddFocus( focusedWindowIdx );
		} else {
			// Если выделение пропало - значит ход завершен убраны все старые выделения.
			focusedWindowIdx = -1;
		}
	}
}

void CFieldWindow::OnKillFocus() const
{
	if( focusedWindowIdx == windowField.Name ) {
		focusedWindowIdx = -1;
	}
	engine.DelFocus( windowField.Name );
}

LRESULT __stdcall CFieldWindow::fieldWindowProc( HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam ) 
{
	CFieldWindow* window = reinterpret_cast<CFieldWindow*>( ::GetWindowLongPtr( hwnd, GWLP_USERDATA ) );
	switch( message ) {
		case WM_NCCREATE:
			::SetWindowLongPtr( hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>( reinterpret_cast<CREATESTRUCT*>( lParam )->lpCreateParams ) );
			return 1;
		case WM_DESTROY:
			window->OnDestroy();
			break;
		case WM_PAINT:
			window->OnPaint();
			break;
		case WM_LBUTTONDOWN:
			window->OnLButtonDown();
			break;
		case WM_KILLFOCUS:
			window->OnKillFocus();
			break;
		default:
			return ::DefWindowProc( hwnd, message, wParam, lParam );
			break;
	}
	return 0;
}