#-------------------------------------------------
#
# Project created by QtCreator 2014-08-15T17:14:09
#
#-------------------------------------------------

QT += core gui xml

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = ConfigModule
TEMPLATE = app

SOURCES += \
    scaninterfacefunction.cpp \
    main.cpp

HEADERS  += \
    scaninterfacefunction.h

FORMS    += \
    scaninterfacefunction.ui

OTHER_FILES += \
    RobotSDK_Main.pri \
    RobotSDK_Install.pri

unix{
    DESTDIR = $$(HOME)/Build/RobotSDK/Tools/ConfigModule

	MOC_DIR = $$(HOME)/Build/RobotSDK/Tools/ConfigModule/MOC
    OBJECTS_DIR = $$(HOME)/Build/RobotSDK/Tools/ConfigModule/OBJ
    UI_DIR = $$(HOME)/Build/RobotSDK/Tools/ConfigModule/UI
	
    target.path = $$(HOME)/SDK/RobotSDK/Tools
    INSTALLS += target

    PUBLISHFILES.path = $$(HOME)/SDK/RobotSDK/Tools
    PUBLISHFILES.files = $$OTHER_FILES
    INSTALLS += PUBLISHFILES
}

win32{
    DESTDIR = $$(RobotSDK_Tools)/../../../Build/RobotSDK/Tools/ConfigModule

	MOC_DIR = $$(RobotSDK_Kernel)/../../../Build/RobotSDK/Tools/ConfigModule/MOC
    OBJECTS_DIR = $$(RobotSDK_Kernel)/../../../Build/RobotSDK/Tools/ConfigModule/OBJ
    UI_DIR = $$(RobotSDK_Kernel)/../../../Build/RobotSDK/Tools/ConfigModule/UI
	
    target.path = $$(RobotSDK_Tools)
    INSTALLS += target

    PUBLISHFILES.path = $$(RobotSDK_Tools)
    PUBLISHFILES.files = $$OTHER_FILES
    INSTALLS += PUBLISHFILES
}
