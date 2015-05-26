#!/bin/sh
TMPBASEDIR=$PWD;
sudo apt-get -y install qt5-default graphviz-dev doxygen libeigen3-dev;
cd $TMPBASEDIR/Src/Doc;
mkdir -p $HOME/SDK/RobotSDK_4.0/Doc;
doxygen RobotSDK_Linux.doc;
ln -s $HOME/SDK/RobotSDK_4.0/Doc/html/index.html $HOME/SDK/RobotSDK_4.0/Doc/RobotSDK.html
cd $TMPBASEDIR;
qmake -makefile $TMPBASEDIR/Src/Kernel/Kernel.pro -r -o "Makefile.Release" "CONFIG+=release";
make -f Makefile.Release;
make -f Makefile.Release install;
rm Makefile.Release;
qmake -makefile $TMPBASEDIR/Src/Kernel/Kernel.pro -r -o "Makefile.Debug" "CONFIG+=debug";
make -f Makefile.Debug;
make -f Makefile.Debug install;
rm Makefile.Debug;
if ! grep -Fxq "source $HOME/SDK/RobotSDK_4.0/Kernel/setup.bash" ~/.bashrc;
then echo "source $HOME/SDK/RobotSDK_4.0/Kernel/setup.bash" >> ~/.bashrc;
fi;
. $HOME/SDK/RobotSDK_4.0/Kernel/setup.bash;
qmake -makefile $TMPBASEDIR/Src/Robot-X/Robot-X.pro -r -o "Makefile.Release" "CONFIG+=release";
make -f Makefile.Release -B;
make -f Makefile.Release install;
rm Makefile.Release;
qmake -makefile $TMPBASEDIR/Src/Robot-X/Robot-X.pro -r -o "Makefile.Debug" "CONFIG+=debug";
make -f Makefile.Debug -B;
make -f Makefile.Debug install;
rm Makefile.Debug;
