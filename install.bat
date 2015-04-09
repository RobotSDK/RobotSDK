@echo off

set "TMPCURPATH=%cd%"
set "TMPBATPATH=%~dp0"

set "TMPDISKDRIVER="
set "QTDIR="
set "BOOSTDIR="

if defined RobotSDK_Kernel set "TMPDISKDRIVER=%RobotSDK_Kernel:~0,1%" & goto ConfigRobotSDK
if not "%1"=="" set "TMPDISKDRIVER=%1" & goto ConfigRobotSDK
set /p TMPDISKDRIVER=Input disk driver for RobotSDK (default c):

:ConfigRobotSDK

set "RobotSDK_Kernel=%TMPDISKDRIVER%:\SDK\RobotSDK\Kernel"
set "RobotSDK_Tools=%TMPDISKDRIVER%:\SDK\RobotSDK\Tools"

:ConfigQt
 
if not "%2"=="" set "QTDIR=%2"
if not "%3"=="" set "BOOSTDIR=%3"
if defined RobotDep_Bin set "PATH=%PATH%;%RobotDep_Bin%;" & goto InstallRobotSDK

if defined QTDIR goto ConfigBoost
set /p QTDIR=Input path of Qt (contains bin, include and lib):

:ConfigBoost

if defined BOOSTDIR goto ConfigRobotDep
set /p BOOSTDIR=Input libXX-msvc-XX.X path of Boost:

:ConfigRobotDep

set "RobotDep_Include=%QTDIR%\include;%BOOSTDIR%\..;"
set "RobotDep_Lib=%QTDIR%\lib;%BOOSTDIR%;"
set "RobotDep_Bin=%QTDIR%\bin;%BOOSTDIR%;"
set "PATH=%PATH%;%RobotDep_Bin%;"

:InstallRobotSDK

echo Start Building Tools!

if not exist %TMPDISKDRIVER%:\Build\RobotSDK\Tools\NUL mkdir %TMPDISKDRIVER%:\Build\RobotSDK\Tools
if not exist %TMPDISKDRIVER%:\Build\RobotSDK\VS\Tools\NUL mkdir %TMPDISKDRIVER%:\Build\RobotSDK\VS\Tools
cd /D %TMPDISKDRIVER%:\Build\RobotSDK\VS\Tools

qmake -tp vc -r "%TMPBATPATH:\=/%/Src/Tools/Tools.pro" "CONFIG+=build_all"
qmake -makefile "%TMPBATPATH:\=/%/Src/Tools/Tools.pro" -r "CONFIG+=release"
nmake
nmake install

echo Start ConfigSystem.exe

cd "%TMPDISKDRIVER%:\Build\RobotSDK\Tools"

if defined QTDIR if defined BOOSTDIR start .\ConfigSystem.exe %TMPDISKDRIVER% %QTDIR% %BOOSTDIR%

:FinishConfigSystem

echo Tools Building Completed!

echo Start Building Kernel!

if not exist %TMPDISKDRIVER%:\Build\RobotSDK\Kernel\include\NUL mkdir %TMPDISKDRIVER%:\Build\RobotSDK\Kernel\include
if not exist %TMPDISKDRIVER%:\Build\RobotSDK\Kernel\lib\NUL mkdir %TMPDISKDRIVER%:\Build\RobotSDK\Kernel\lib
if not exist %TMPDISKDRIVER%:\Build\RobotSDK\VS\Kernel\NUL mkdir %TMPDISKDRIVER%:\Build\RobotSDK\VS\Kernel
cd /D "%TMPDISKDRIVER%:\Build\RobotSDK\VS\Kernel"

qmake -tp vc "%TMPBATPATH:\=/%/Src/Kernel/Kernel.pro" "CONFIG+=build_all"
qmake -makefile "%TMPBATPATH:\=/%/Src/Kernel/Kernel.pro" -r "CONFIG+=build_all"
nmake -f Makefile.Release
nmake -f Makefile.Release install
nmake -f Makefile.Debug
nmake -f Makefile.Debug install

echo Kernel Building Completed!

echo Start Generating Documentation!

cd /D "%TMPBATPATH%\Doc\Doxygen"

doxygen RobotSDK

set "KernelPath=%RobotSDK_Kernel:/=\%"
if not exist %TMPBATPATH%\Doc\html\NUL echo Documentation is not compiled! & goto Finish
if not exist %KernelPath%\..\Doc\NUL mkdir %KernelPath%\..\Doc
xcopy %TMPBATPATH%\Doc\html\* %KernelPath%\..\Doc /s /e /y
RMDIR /S /Q %TMPBATPATH%\Doc\html


echo Documentation Generation Completed!

:Finish

echo Installation Completed!

:ExitBat

cd /D %TMPCURPATH%

@echo on
