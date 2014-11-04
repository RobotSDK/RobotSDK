@echo off

set TMPSDKNAME=KITTI
set TMPTARGET=MOD
set TMPPRENAME=KITTI

set TMPCURPATH=%cd%

cd /D %~dp0

if "TMPTARGET"=="SDK" goto BUILDSDK
if "TMPTARGET"=="APP" goto BUILDAPP
if "TMPTARGET"=="MOD" goto BUILDMOD

:BUILDSDK

if not exist .\Build\include\NUL mkdir Build\include
if not exist .\Build\lib\NUL mkdir Build\lib
if not exist .\VS\NUL mkdir VS
cd VS

set PROPATH=../%TMPPRENAME%/%TMPSDKNAME%.pro
qmake -tp vc -r %PROPATH% "CONFIG+=build_all"
devenv %TMPSDKNAME%.vcxproj /build Release
devenv %TMPSDKNAME%.vcxproj /build Debug
xcopy ..\..\Src\*.h ..\..\Build\include\ /s /e /y

goto INSTALL

:BUILDAPP

if not exist .\Build\NUL mkdir Build
if not exist .\VS\NUL mkdir VS
cd VS

set PROPATH=../%TMPPRENAME%/%TMPSDKNAME%.pro
qmake -tp vc -r %PROPATH% "CONFIG+=build_all"
devenv %TMPSDKNAME%.vcxproj /build Release
devenv %TMPSDKNAME%.vcxproj /build Debug

goto INSTALL

:BUILDMOD

if not exist .\Build\NUL mkdir Build
if not exist .\VS\NUL mkdir VS
cd VS

set PROPATH=../%TMPPRENAME%/%TMPSDKNAME%.pro
qmake -tp vc -r %PROPATH% "CONFIG+=build_all"
devenv %TMPSDKNAME%.vcxproj /build Release
devenv %TMPSDKNAME%.vcxproj /build Debug

:INSTALL

echo Building Completed!

set INSTALLPATH=C:\%TMPTARGET%\%TMPSDKNAME%
if not "%1"=="" set INSTALLPATH=%1:\%TMPTARGET%\%TMPSDKNAME%
if "%TMPTARGET%"=="MOD" set INSTALLPATH=%RobotSDK_SharedLibrary%

if not exist %INSTALLPATH%\NUL mkdir %INSTALLPATH%
xcopy .\Build\* %INSTALLPATH% /s /e /y

echo Installation Completed!

cd /D %TMPCURPATH%

pause

@echo on
