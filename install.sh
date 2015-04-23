#!/bin/sh
TMPBASEDIR=$PWD;
sudo apt-get -y install graphviz-dev doxygen libeigen3-dev;
cd $TMPBASEDIR/Doc/Doxygen;
doxygen RobotSDK;
cd $TMPBASEDIR/Doc/html;
mkdir -p $HOME/SDK/RobotSDK_4.0/Doc;
cp * $HOME/SDK/RobotSDK_4.0/Doc;
rm -rf $TMPBASEDIR/Doc/html;
cd $TMPBASEDIR;
qmake-5 -makefile $TMPBASEDIR/Src/Kernel/Kernel.pro -r -o "Makefile.Release" "CONFIG+=release";
make -f Makefile.Release;
make -f Makefile.Release install;
rm Makefile.Release;
qmake-5 -makefile $TMPBASEDIR/Src/Kernel/Kernel.pro -r -o "Makefile.Debug" "CONFIG+=debug";
make -f Makefile.Debug;
make -f Makefile.Debug install;
rm Makefile.Debug;
qmake-5 -makefile $TMPBASEDIR/Src/Robot-X/Robot-X.pro -r -o "Makefile.Release" "CONFIG+=release";
make -f Makefile.Release;
make -f Makefile.Release install;
rm Makefile.Release;
qmake=5 -makefile $TMPBASEDIR/Src/Robot-X/Robot-X.pro -r -o "Makefile.Debug" "CONFIG+=debug";
make -f Makefile.Debug;
make -f Makefile.Debug install;
rm Makefile.Debug;
rm install;

