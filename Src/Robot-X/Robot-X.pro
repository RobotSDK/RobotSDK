#-------------------------------------------------
#
# Project created by QtCreator 2015-04-11T17:52:19
#
#-------------------------------------------------

QT       += core gui xml

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Robot-X
TEMPLATE = app

CONFIG += c++11

unix {
    INCLUDEPATH += $$(HOME)/SDK/RobotSDK_4.0/Kernel/include
    LIBS += -L$$(HOME)/SDK/RobotSDK_4.0/Kernel/lib -lKernel_Debug
}

win32 {
    INCLUDEPATH += c:/SDK/RobotSDK_4.0/Kernel/include
    LIBS += -Lc:/SDK/RobotSDK_4.0/Kernel/lib -lKernel_Debug
}

SOURCES += main.cpp\
        mainwindow.cpp \
    xnode.cpp \
    xedge.cpp \
    xport.cpp \
    xgraph.cpp

HEADERS  += mainwindow.h \
    xnode.h \
    xedge.h \
    xport.h \
    xgraph.h

FORMS    += mainwindow.ui

INCLUDEPATH += $(HOME)/
