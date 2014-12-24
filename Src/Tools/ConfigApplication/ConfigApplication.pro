#-------------------------------------------------
#
# Project created by QtCreator 2014-08-15T17:14:55
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = ConfigApplication
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h

FORMS    += mainwindow.ui

unix{
    DESTDIR = $$(HOME)/Build/RobotSDK/Tools/ConfigApplication
	
    MOC_DIR = $$(HOME)/Build/RobotSDK/Tools/ConfigApplication/MOC
    OBJECTS_DIR = $$(HOME)/Build/RobotSDK/Tools/ConfigApplication/OBJ
    UI_DIR = $$(HOME)/Build/RobotSDK/Tools/ConfigApplication/UI

    target.path = $$(HOME)/SDK/RobotSDK/Tools
    INSTALLS += target
}

win32{
    DESTDIR = $$(RobotSDK_Tools)/../../../Build/RobotSDK/Tools/ConfigApplication

    MOC_DIR = $$(RobotSDK_Kernel)/../../../Build/RobotSDK/Tools/ConfigApplication/MOC
    OBJECTS_DIR = $$(RobotSDK_Kernel)/../../../Build/RobotSDK/Tools/ConfigApplication/OBJ
    UI_DIR = $$(RobotSDK_Kernel)/../../../Build/RobotSDK/Tools/ConfigApplication/UI
	
    target.path = $$(RobotSDK_Tools)
    INSTALLS += target
}
