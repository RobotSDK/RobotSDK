#RobotSDK configurations
export ROBOTSDKVER=4.0
export ROBOTSDKMODULE=$HOME/SDK/RobotSDK_$ROBOTSDKVER/Kernel/RobotSDK.pri
export ROBOTSDKCUDA=$HOME/SDK/RobotSDK_$ROBOTSDKVER/Kernel/RobotSDK_CUDA.pri
ROBOTXDIR=$HOME/SDK/RobotSDK_$ROBOTSDKVER/Robot-X

#Additional environment settings (seperated by ':')
if ! test -f $HOME/SDK/RobotSDK_$ROBOTSDKVER/Kernel/RobotSDK_Env.bash
then
    echo "#Additional environment settings (seperated by ':')" >> $HOME/SDK/RobotSDK_$ROBOTSDKVER/Kernel/RobotSDK_Env.bash
    echo "ADDPATH=" >> $HOME/SDK/RobotSDK_$ROBOTSDKVER/Kernel/RobotSDK_Env.bash
    echo "ADDLIBRARY=" >> $HOME/SDK/RobotSDK_$ROBOTSDKVER/Kernel/RobotSDK_Env.bash
fi
source $HOME/SDK/RobotSDK_$ROBOTSDKVER/Kernel/RobotSDK_Env.bash

#Export environment settings
export RobotSDKDIR=$HOME/SDK/RobotSDK_$ROBOTSDKVER
if test -z "$ADDPATH"
then
    ADDPATH=$ROBOTXDIR
else
    ADDPATH=$ROBOTXDIR:$ADDPATH
fi
if test -z "$ADDLIBRARY"
then
    ADDLIBRARY=/opt/ros/$ROS_DISTRO/lib:/usr/lib/x86_64-linux-gnu
else
    ADDLIBRARY=/opt/ros/$ROS_DISTRO/lib:/usr/lib/x86_64-linux-gnu:$ADDLIBRARY
fi
export PATH=$ADDPATH:$PATH
export LD_LIBRARY_PATH=$ADDLIBRARY:$LD_LIBRARY_PATH
