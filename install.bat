@echo off

set "TMPCURPATH=%cd%"
set "TMPBATPATH=%~dp0"
set "ROBOTSDKDIR=C:\SDK\RobotSDK_4.0"

echo Start Generating Documentation!

cd /D "%TMPBATPATH%\Src\Doc"

doxygen RobotSDK_Windows.doc

if not exist %ROBOTSDKDIR%\Doc\html\NUL echo Documentation is not compiled! & goto InstallRobotSDK

echo Documentation Generation Completed!

:InstallRobotSDK

echo Start Building RobotSDK!

if not exist %ROBOTSDKDIR%\Build\Kernel\VS\NUL mkdir %ROBOTSDKDIR%\Build\Kernel\VS
cd /D "%ROBOTSDKDIR%\Build\Kernel\VS"

qmake -tp vc "%TMPBATPATH:\=/%/Src/Kernel/Kernel.pro" "CONFIG+=build_all"
qmake -makefile "%TMPBATPATH:\=/%/Src/Kernel/Kernel.pro" -r "CONFIG+=build_all"
nmake -f Makefile.Release
nmake -f Makefile.Release install
nmake -f Makefile.Debug
nmake -f Makefile.Debug install

echo Installation Completed!

cd /D %TMPCURPATH%

@echo on
