#!/bin/sh
SDKNAME=KITTI;
PRENAME=KITTI;
sudo apt-get -y install libeigen3-dev;
mkdir -p GCC;cd GCC;
qmake -makefile ../$PRENAME/$SDKNAME.pro -o Makefile.release -r "CONFIG+=release";
make -f Makefile.release;
make -f Makefile.release install;
qmake -makefile ../$PRENAME/$SDKNAME.pro -o Makefile.debug -r "CONFIG+=debug";
make -f Makefile.debug;
make -f Makefile.debug install;
