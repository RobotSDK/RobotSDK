#-------------------------------------------------
#
# Project created by QtCreator 2014-08-15T17:14:33
#
#-------------------------------------------------

QT       += core gui xml

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = ConfigProject
TEMPLATE = app


SOURCES += \
    configproject.cpp \
    main.cpp

HEADERS  += \
    configproject.h

FORMS    += \
    configproject.ui

unix{
    DESTDIR = $$(HOME)/Build/RobotSDK/Tools/ConfigProject

	MOC_DIR = $$(HOME)/Build/RobotSDK/Tools/ConfigProject/MOC
    OBJECTS_DIR = $$(HOME)/Build/RobotSDK/Tools/ConfigProject/OBJ
    UI_DIR = $$(HOME)/Build/RobotSDK/Tools/ConfigProject/UI
	
    target.path = $$(HOME)/SDK/RobotSDK/Tools
    INSTALLS += target
}

win32{
    DESTDIR = $$(RobotSDK_Tools)/../../../Build/RobotSDK/Tools/ConfigProject

	MOC_DIR = $$(RobotSDK_Kernel)/../../../Build/RobotSDK/Tools/ConfigProject/MOC
    OBJECTS_DIR = $$(RobotSDK_Kernel)/../../../Build/RobotSDK/Tools/ConfigProject/OBJ
    UI_DIR = $$(RobotSDK_Kernel)/../../../Build/RobotSDK/Tools/ConfigProject/UI
	
    target.path = $$(RobotSDK_Tools)
    INSTALLS += target
}
