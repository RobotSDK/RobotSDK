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

unix{
    DESTDIR = $$(HOME)/Build/RobotSDK/Tools/ConfigModule

    MOC_DIR = $$(HOME)/Build/RobotSDK/Tools/ConfigModule/MOC
    OBJECTS_DIR = $$(HOME)/Build/RobotSDK/Tools/ConfigModule/OBJ
    UI_DIR = $$(HOME)/Build/RobotSDK/Tools/ConfigModule/UI
	
    target.path = $$(HOME)/SDK/RobotSDK/Tools
    INSTALLS += target
}

win32{
    DESTDIR = $$(RobotSDK_Tools)/../../../Build/RobotSDK/Tools/ConfigModule

    MOC_DIR = $$(RobotSDK_Kernel)/../../../Build/RobotSDK/Tools/ConfigModule/MOC
    OBJECTS_DIR = $$(RobotSDK_Kernel)/../../../Build/RobotSDK/Tools/ConfigModule/OBJ
    UI_DIR = $$(RobotSDK_Kernel)/../../../Build/RobotSDK/Tools/ConfigModule/UI
	
    target.path = $$(RobotSDK_Tools)
    INSTALLS += target
}
