#-------------------------------------------------
#
# Project created by QtCreator 2015-04-10T15:08:03
#
#-------------------------------------------------

QT       += core gui xml

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = TestGraph
TEMPLATE = app

CONFIG += c++11

DEFINES += RobotSDK_Application
INCLUDEPATH += $$(HOME)/SDK/RobotSDK_4.0/Kernel/include
LIBS += -L$$(HOME)/SDK/RobotSDK_4.0/Kernel/lib -lKernel_Debug

SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h

FORMS    += mainwindow.ui
