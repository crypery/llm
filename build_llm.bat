@echo off
setlocal
set ROOT=%~dp0
if "%ROOT:~-1%"=="\" set ROOT=%ROOT:~0,-1%
set GGML=%ROOT%\ggml

if /i "%~1"=="all" (
    call "%~f0" cpu || exit /b 1
    call "%~f0" gpu || exit /b 1
    echo done: %ROOT%\test_llm_cpu.exe + %ROOT%\test_llm_gpu.exe
    exit /b 0
)

if /i "%~1"=="cpu" goto :mode_cpu
if /i "%~1"=="gpu" goto :mode_gpu
goto :mode_default

:mode_cpu
set VULKAN_LIB=
set TAG=_cpu
set OUT=test_llm_cpu.exe
goto :mode_done

:mode_gpu
if not defined VULKAN_SDK (
    echo VULKAN_SDK is not set, cannot build the GPU binary
    exit /b 1
)
set VULKAN_LIB=%VULKAN_SDK%\Lib\vulkan-1.lib
if not exist "%VULKAN_LIB%" (
    echo not found: %VULKAN_LIB%
    exit /b 1
)
set TAG=_gpu
set OUT=test_llm_gpu.exe
goto :mode_done

:mode_default
:: default: historical behavior (test_llm.exe; Vulkan only if VULKAN_LIB preset)
::set VULKAN_LIB=%VULKAN_SDK%\Lib\vulkan-1.lib
if not exist "%VULKAN_LIB%" set VULKAN_LIB=
set TAG=
set OUT=test_llm.exe

:mode_done

if "%VULKAN_LIB%"=="" (
    set BUILD=%GGML%\build-mingw
) else (
    set BUILD=%GGML%\build-mingw-vk
)

if not exist "%BUILD%\src\ggml-cpu.a" (
    echo [1/3] configuring ggml [MinGW Makefiles]...
    if "%VULKAN_LIB%"=="" (
        cmake -S "%GGML%" -B "%BUILD%" -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF -DGGML_CCACHE=OFF -DGGML_STATIC=ON || exit /b 1
    ) else (
        cmake -S "%GGML%" -B "%BUILD%" -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF -DGGML_CCACHE=OFF -DGGML_STATIC=ON -DGGML_VULKAN=ON || exit /b 1
    )
    echo [2/3] building ggml...
    cmake --build "%BUILD%" --config Release -j 8 || exit /b 1
) else (
    echo [1/3] ggml already built
)

echo [2/3] compiling common.c + llm.c + test_llm.c + tools.c + log.c [%OUT%]...
if "%VULKAN_LIB%"=="" (
    set VKDEF=
) else (
    set VKDEF=-DGGML_USE_VULKAN
)
gcc -O2 %VKDEF% -I "%ROOT%" -I "%GGML%\include" -I "%GGML%\src" -c "%ROOT%\common.c" -o "%ROOT%\common%TAG%.o" || exit /b 1
gcc -O2 %VKDEF% -mavx2 -mfma -fopenmp -I "%ROOT%" -I "%GGML%\include" -I "%GGML%\src" -c "%ROOT%\llm.c" -o "%ROOT%\llm%TAG%.o" || exit /b 1
gcc -O2 %VKDEF% -I "%ROOT%" -I "%GGML%\include" -I "%GGML%\src" -c "%ROOT%\test_llm.c" -o "%ROOT%\test_llm%TAG%.o" || exit /b 1
gcc -O2 -I "%ROOT%" -c "%ROOT%\tools.c" -o "%ROOT%\tools%TAG%.o" || exit /b 1
gcc -O2 %VKDEF% -I "%ROOT%" -I "%GGML%\include" -I "%GGML%\src" -c "%ROOT%\log.c" -o "%ROOT%\log%TAG%.o" || exit /b 1

echo [3/3] linking binary...
if "%VULKAN_LIB%"=="" (
    g++ "%ROOT%\test_llm%TAG%.o" "%ROOT%\llm%TAG%.o" "%ROOT%\common%TAG%.o" "%ROOT%\log%TAG%.o" "%ROOT%\tools%TAG%.o" -Wl,--start-group "%BUILD%\src\ggml-cpu.a" "%BUILD%\src\ggml.a" "%BUILD%\src\ggml-base.a" -Wl,--end-group -o "%ROOT%\%OUT%" -pthread -fopenmp -lole32 || exit /b 1
) else (
    g++ "%ROOT%\test_llm%TAG%.o" "%ROOT%\llm%TAG%.o" "%ROOT%\common%TAG%.o" "%ROOT%\log%TAG%.o" "%ROOT%\tools%TAG%.o" -Wl,--start-group "%BUILD%\src\ggml-vulkan\ggml-vulkan.a" "%BUILD%\src\ggml-cpu.a" "%BUILD%\src\ggml.a" "%BUILD%\src\ggml-base.a" -Wl,--end-group "%VULKAN_LIB%" -o "%ROOT%\%OUT%" -pthread -fopenmp -lole32 || exit /b 1
)
echo done: %ROOT%\%OUT%
