@echo off

set "TMPCURPATH=%cd%"
set "TMPBATPATH=%~dp0"

set "TMPDISKDRIVER="

if not "%1"=="" set "TMPDISKDRIVER=%1" & goto ConfigRobotSDK
set /p TMPDISKDRIVER=Input disk driver for RobotSDK (default c):

:ConfigRobotSDK

set "ROBOTSDKDIR=%TMPDISKDRIVER%:\SDK\RobotSDK_4.0"

:InstallRobotSDK

echo Start Building Kernel!

if not exist %ROBOTSDKDIR%\Build\Kernel\VS\NUL mkdir %ROBOTSDKDIR%\Build\Kernel\VS
cd /D "%ROBOTSDKDIR%\Build\Kernel\VS"

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

if not exist %TMPBATPATH%\Doc\html\NUL echo Documentation is not compiled! & goto Finish
if not exist %ROBOTSDKDIR%\Doc\NUL mkdir %ROBOTSDKDIR%\Doc
xcopy %TMPBATPATH%\Doc\html\* %ROBOTSDKDIR%\Doc /s /e /y
RMDIR /S /Q %TMPBATPATH%\Doc\html

echo Documentation Generation Completed!

:Finish

echo Installation Completed!

:ExitBat

cd /D %TMPCURPATH%

@echo on
