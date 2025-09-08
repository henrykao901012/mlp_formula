@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

echo ==========================================
echo       Conda 環境重建工具
echo ==========================================
echo.

:: 檢查是否有傳入參數（檔案路徑）
if "%~1"=="" (
    echo 使用方式：
    echo   拖拽 yml 或 txt 檔案到此 bat 檔案上
    echo   或者: %~nx0 檔案路徑
    echo.
    echo 支援的檔案類型：
    echo   - environment.yml ^(conda 環境檔^)
    echo   - requirements.txt ^(pip 需求檔^)
    echo.
    pause
    exit /b 1
)

:: 取得檔案路徑和副檔名
set "filepath=%~1"
set "filename=%~n1"
set "extension=%~x1"

echo 檔案路徑: %filepath%
echo 檔案名稱: %filename%
echo 副檔名: %extension%
echo.

:: 檢查檔案是否存在
if not exist "%filepath%" (
    echo 錯誤：找不到檔案 "%filepath%"
    pause
    exit /b 1
)

:: 根據副檔名決定處理方式
if /i "%extension%"==".yml" (
    goto :handle_yml
) else if /i "%extension%"==".yaml" (
    goto :handle_yml
) else if /i "%extension%"==".txt" (
    goto :handle_txt
) else (
    echo 錯誤：不支援的檔案類型 "%extension%"
    echo 僅支援 .yml, .yaml, .txt 檔案
    pause
    exit /b 1
)

:handle_yml
echo 偵測到 YAML 環境檔案
echo.

:: 詢問環境名稱
set /p "env_name=請輸入新環境名稱 (留空使用檔案中的名稱): "

echo.
echo 準備執行的命令：
if "%env_name%"=="" (
    echo conda env create -f "%filepath%"
    echo.
    set /p "confirm=確認執行？ (y/N): "
    if /i "!confirm!"=="y" (
        conda env create -f "%filepath%"
    ) else (
        echo 取消操作
        goto :end
    )
) else (
    echo conda env create -n %env_name% -f "%filepath%"
    echo.
    set /p "confirm=確認執行？ (y/N): "
    if /i "!confirm!"=="y" (
        conda env create -n %env_name% -f "%filepath%"
    ) else (
        echo 取消操作
        goto :end
    )
)
goto :end

:handle_txt
echo 偵測到文字需求檔案
echo.

:: 詢問環境名稱
set /p "env_name=請輸入新環境名稱: "
if "%env_name%"=="" (
    echo 錯誤：環境名稱不能為空
    pause
    exit /b 1
)

echo.
echo 選擇安裝方式：
echo [1] 使用 conda 安裝 (conda install)
echo [2] 使用 pip 安裝 (pip install)
echo [3] 建立基礎環境後用 pip 安裝

set /p "choice=請選擇 (1-3): "

if "%choice%"=="1" (
    echo.
    echo 準備執行的命令：
    echo conda create -n %env_name% --file "%filepath%"
    echo.
    set /p "confirm=確認執行？ (y/N): "
    if /i "!confirm!"=="y" (
        conda create -n %env_name% --file "%filepath%"
    ) else (
        echo 取消操作
        goto :end
    )
) else if "%choice%"=="2" (
    echo.
    echo 準備執行的命令：
    echo conda create -n %env_name% python
    echo conda activate %env_name%
    echo pip install -r "%filepath%"
    echo.
    set /p "confirm=確認執行？ (y/N): "
    if /i "!confirm!"=="y" (
        conda create -n %env_name% python -y
        call conda activate %env_name%
        pip install -r "%filepath%"
    ) else (
        echo 取消操作
        goto :end
    )
) else if "%choice%"=="3" (
    echo.
    set /p "python_ver=請輸入 Python 版本 (例如 3.9, 留空使用預設): "
    echo.
    echo 準備執行的命令：
    if "%python_ver%"=="" (
        echo conda create -n %env_name% python
    ) else (
        echo conda create -n %env_name% python=%python_ver%
    )
    echo conda activate %env_name%
    echo pip install -r "%filepath%"
    echo.
    set /p "confirm=確認執行？ (y/N): "
    if /i "!confirm!"=="y" (
        if "%python_ver%"=="" (
            conda create -n %env_name% python -y
        ) else (
            conda create -n %env_name% python=%python_ver% -y
        )
        call conda activate %env_name%
        pip install -r "%filepath%"
    ) else (
        echo 取消操作
        goto :end
    )
) else (
    echo 無效的選擇
    goto :end
)

:end
echo.
if %ERRORLEVEL% EQU 0 (
    echo ==========================================
    echo 環境重建完成！
    echo.
    echo 啟用環境請使用：
    if not "%env_name%"=="" (
        echo conda activate %env_name%
    )
    echo ==========================================
) else (
    echo ==========================================
    echo 環境重建過程中發生錯誤！
    echo 錯誤代碼: %ERRORLEVEL%
    echo ==========================================
)

echo.
pause