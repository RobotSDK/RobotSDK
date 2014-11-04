#-------------------------------------------------
#
# Project created by QtCreator 2014-10-09T13:30:41
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Publish
TEMPLATE = app

SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h

FORMS    += mainwindow.ui

unix{
    DESTDIR = $$(HOME)/Build/RobotSDK/Tools

    target.path = $$(HOME)/SDK/RobotSDK/Tools
    INSTALLS += target
}

win32{
    DESTDIR = $$(RobotSDK_Tools)/../../../Build/RobotSDK/Tools

    target.path = $$(RobotSDK_Tools)
    INSTALLS += target
}