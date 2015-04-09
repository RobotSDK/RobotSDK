#!/bin/sh
TMPBASEDIR=$PWD;
qmake -makefile $TMPBASEDIR/Src/Kernel/Kernel.pro -r -o "Makefile.Release" "CONFIG+=release";
make -f Makefile.Release;
make -f Makefile.Release install;
rm Makefile.Release;
qmake -makefile $TMPBASEDIR/Src/Kernel/Kernel.pro -r -o "Makefile.Debug" "CONFIG+=debug";
make -f Makefile.Debug;
make -f Makefile.Debug install;
rm Makefile.Debug;
cd $TMPBASEDIR/Doc/Doxygen;
doxygen RobotSDK;
cd $TMPBASEDIR/Doc/html;
mkdir -p $HOME/SDK/RobotSDK_4.0/Doc;
cp * $HOME/SDK/RobotSDK_4.0/Doc;
rm -rf $TMPBASEDIR/Doc/html;
