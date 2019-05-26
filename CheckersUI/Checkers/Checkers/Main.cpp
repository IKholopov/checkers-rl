// Автор: Фролов Николай.

#include <MainWindow.h>
#include <FieldWindow.h>

#include <Windows.h>

int _stdcall wWinMain( HINSTANCE hInstance, HINSTANCE prevInstance, LPWSTR commandLine, int nCmdShow )
{
	CFieldWindow::RegisterClass();
	CMainWindow::RegisterClass();
    CMainWindow mainWindow;
    mainWindow.Create();
    mainWindow.Show( nCmdShow );
    MSG msg;
    while( ::GetMessage( &msg, 0, 0, 0 ) != 0 ) {
        ::TranslateMessage( &msg );
        ::DispatchMessage( &msg );
    }
	return 0;
}