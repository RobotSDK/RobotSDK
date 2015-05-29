#Additional environment settings (seperated by ':')
if ! test -f $HOME/SDK/RobotSDK_4.0/Kernel/RobotSDK_Env.bash
then
    echo "#Additional environment settings (seperated by ':')" >> $HOME/SDK/RobotSDK_4.0/Kernel/RobotSDK_Env.bash
    echo "ADDPATH=" >> $HOME/SDK/RobotSDK_4.0/Kernel/RobotSDK_Env.bash
    echo "ADDLIBRARY=" >> $HOME/SDK/RobotSDK_4.0/Kernel/RobotSDK_Env.bash
fi
source $HOME/SDK/RobotSDK_4.0/Kernel/RobotSDK_Env.bash

#QT configurations
QTVER=5.4
QTCOMPILER=gcc_64
QTDIR=$HOME/SDK/Qt
QTMDIR=$QTDIR/$QTVER/$QTCOMPILER
QTCDIR=$QTDIR/Tools/QtCreator

#Export environment settings
export RobotSDKDIR=$HOME/SDK/RobotSDK_4.0
if test -z "$ADDPATH"
then
    ADDPATH=$QTMDIR/bin:$QTCDIR/bin:$HOME/SDK/RobotSDK_4.0/Robot-X
else
    ADDPATH=$QTMDIR/bin:$QTCDIR/bin:$HOME/SDK/RobotSDK_4.0/Robot-X:$ADDPATH
fi
if test -z "$ADDLIBRARY"
then
    ADDLIBRARY=$QTDIR/lib:$QTCDIR/lib/qtcreator:/opt/ros/$ROS_DISTRO/lib
else
    ADDLIBRARY=$QTDIR/lib:$QTCDIR/lib/qtcreator:/opt/ros/$ROS_DISTRO/lib:$ADDLIBRARY
fi
export PATH=$ADDPATH:$PATH
export LD_LIBRARY_PATH=$ADDLIBRARY:$LD_LIBRARY_PATH
