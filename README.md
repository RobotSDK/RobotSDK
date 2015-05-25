RobotSDK_4.0
========
How to install RobotSDK 

1 Get source code 
```
git clone https://github.com/RobotSDK/RobotSDK.git
```
2 Install RobotSDK

  (1) for Linux
```
cd RobotSDK
sh Install.sh

```
 Setup environment variables, write them into ~/.bashrc:
```
QTVER=5.4
QTDIR=$HOME/SDK/Qt/$QTVER/gcc_64
QTCDIR=$HOME/SDK/Qt/Tools/QtCreator
export PATH=$QTDIR/bin:$QTCDIR/bin:$PATH
export LD_LIBRARY_PATH=$QTDIR/lib:$QTCDIR/lib/qtcreator:/opt/ros/$ROS_DISTRO/lib:$LD_LIBRARY_PATH
```

  (2) for Windows (use Visual Studio Command Prompt)
```
cd RobotSDK
Install.bat
```
 Setup environment variables:
 Add Qt and Graphviz to %Path%
