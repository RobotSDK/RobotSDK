#-------------------------------------------------
#
# Project created by QtCreator 2014-08-15T17:09:44
#
#-------------------------------------------------

QT       += core gui xml

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = ConfigSystem
TEMPLATE = app
CONFIG += qt

SOURCES += main.cpp\
    configuration.cpp \
    registerdirwidget.cpp \
    registerdirwidgetitem.cpp

HEADERS  += \
    configuration.h \
    registerdirwidget.h \
    registerdirwidgetitem.h

FORMS    += mainwindow.ui \
    configuration.ui \
    projectsetting.ui \
    registerdirwidget.ui \
    registerdirwidgetitem.ui

unix{
    DESTDIR = $$(HOME)/Build/RobotSDK/Tools/ConfigSystem

	MOC_DIR = $$(HOME)/Build/RobotSDK/Tools/ConfigSystem/MOC
    OBJECTS_DIR = $$(HOME)/Build/RobotSDK/Tools/ConfigSystem/OBJ
    UI_DIR = $$(HOME)/Build/RobotSDK/Tools/ConfigSystem/UI
	
    target.path = $$(HOME)/SDK/RobotSDK/Tools
    INSTALLS += target
}

win32{
    DESTDIR = $$(RobotSDK_Tools)/../../../Build/RobotSDK/Tools/ConfigSystem

	MOC_DIR = $$(RobotSDK_Kernel)/../../../Build/RobotSDK/Tools/ConfigSystem/MOC
    OBJECTS_DIR = $$(RobotSDK_Kernel)/../../../Build/RobotSDK/Tools/ConfigSystem/OBJ
    UI_DIR = $$(RobotSDK_Kernel)/../../../Build/RobotSDK/Tools/ConfigSystem/UI
	
    target.path = $$(RobotSDK_Tools)
    INSTALLS += target

    QMAKE_LFLAGS += /MANIFESTUAC:"level='requireAdministrator'uiAccess='false'"
}


