RobotSDK_4.0 in Jetson TK1 embeded PC
========
Requirements

1. L4T ubuntu14.04 for jetson tk1, I am using r21.3.

2. Compile qt5.3+ yourself, directly install qt5-default wont work.

3. 
```
ln -s <ur qmake> /usr/bin/qmake-5
```
4. Examples contain bugs, someone may fix it later...(maybe)
5. other bugs? 

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
  (2) for Windows (use Visual Studio Command Prompt)
```
cd RobotSDK
Install.bat
```
