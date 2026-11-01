@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if errorlevel 1 goto folder_error

set "BASE_PYTHON="
py -3 -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 set "BASE_PYTHON=py -3"

if not defined BASE_PYTHON (
    python -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)" >nul 2>&1
    if not errorlevel 1 set "BASE_PYTHON=python"
)

if not defined BASE_PYTHON goto python_missing

set "VENV_PYTHON=.tichubot-venv\Scripts\python.exe"
if not exist "%VENV_PYTHON%" (
    echo Preparing TichuBot for its first run...
    %BASE_PYTHON% -m venv ".tichubot-venv"
    if errorlevel 1 goto venv_failed
)

"%VENV_PYTHON%" run_tichubot.py --check-dependencies
if not errorlevel 1 goto start_app

echo.
echo Installing the required CPU packages. Internet access is needed for this first-time step.
"%VENV_PYTHON%" -m pip install --disable-pip-version-check --requirement requirements-windows.txt
if errorlevel 1 goto numpy_failed

"%VENV_PYTHON%" -m pip install --disable-pip-version-check --index-url https://download.pytorch.org/whl/cpu "torch>=2.0"
if errorlevel 1 goto torch_failed

"%VENV_PYTHON%" run_tichubot.py --check-dependencies
if errorlevel 1 goto dependency_failed

:start_app
echo.
echo Starting TichuBot...
"%VENV_PYTHON%" run_tichubot.py
set "APP_STATUS=%ERRORLEVEL%"
if not "%APP_STATUS%"=="0" pause
exit /b %APP_STATUS%

:folder_error
echo.
echo TichuBot could not open its extracted folder.
goto fail

:python_missing
echo.
echo TichuBot requires Python 3.10 or newer.
echo Install Python from https://www.python.org/downloads/windows/
echo During installation, enable "Add python.exe to PATH" or install the Python Launcher.
goto fail

:venv_failed
echo.
echo Python could not create TichuBot's private virtual environment.
goto fail

:numpy_failed
echo.
echo NumPy could not be installed. Check the messages above and your internet connection.
goto fail

:torch_failed
echo.
echo CPU-only PyTorch could not be installed.
echo Check whether your Python version and processor architecture are supported.
goto fail

:dependency_failed
echo.
echo The dependency check still fails after installation.
echo Delete the .tichubot-venv folder and run this launcher again.
goto fail

:fail
echo.
pause
exit /b 1
