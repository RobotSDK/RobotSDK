#-------------------------------------------------
#
# Project created by QtCreator 2015-04-09T23:10:09
#
#-------------------------------------------------

QT       += widgets xml

QT       -= gui

TARGET = TestModule
TEMPLATE = lib

CONFIG += c++11

DEFINES += RobotSDK_Module
INCLUDEPATH += $$(HOME)/SDK/RobotSDK_4.0/Kernel/include
LIBS += -L$$(HOME)/SDK/RobotSDK_4.0/Kernel/lib -lKernel_Debug

SOURCES += testmodule.cpp

HEADERS += testmodule.h

unix {
    target.path = /usr/lib
    INSTALLS += target
}

