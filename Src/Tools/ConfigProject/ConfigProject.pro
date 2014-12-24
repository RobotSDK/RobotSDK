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

OTHER_FILES += \
    Install.sh\
    RobotSDK_Main.pri \
    RobotSDK_Install.pri

unix{
    DESTDIR = $$(HOME)/Build/RobotSDK/Tools/ConfigProject

    MOC_DIR = $$(HOME)/Build/RobotSDK/Tools/ConfigProject/MOC
    OBJECTS_DIR = $$(HOME)/Build/RobotSDK/Tools/ConfigProject/OBJ
    UI_DIR = $$(HOME)/Build/RobotSDK/Tools/ConfigProject/UI
	
    target.path = $$(HOME)/SDK/RobotSDK/Tools
    INSTALLS += target

    PUBLISHFILES.path = $$(HOME)/SDK/RobotSDK/Tools
    PUBLISHFILES.files = $$OTHER_FILES
    INSTALLS += PUBLISHFILES
}

win32{
    DESTDIR = $$(RobotSDK_Tools)/../../../Build/RobotSDK/Tools/ConfigProject

    MOC_DIR = $$(RobotSDK_Kernel)/../../../Build/RobotSDK/Tools/ConfigProject/MOC
    OBJECTS_DIR = $$(RobotSDK_Kernel)/../../../Build/RobotSDK/Tools/ConfigProject/OBJ
    UI_DIR = $$(RobotSDK_Kernel)/../../../Build/RobotSDK/Tools/ConfigProject/UI
	
    target.path = $$(RobotSDK_Tools)
    INSTALLS += target

    PUBLISHFILES.path = $$(RobotSDK_Tools)
    PUBLISHFILES.files = $$OTHER_FILES
    INSTALLS += PUBLISHFILES
}
