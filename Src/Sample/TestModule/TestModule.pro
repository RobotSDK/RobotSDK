#-------------------------------------------------
#
# Project created by QtCreator 2015-04-09T23:10:09
#
#-------------------------------------------------

QT       += widgets

QT       -= gui

TARGET = TestModule
TEMPLATE = lib



unix {
    INCLUDEPATH += $$(HOME)/SDK/RobotSDK_4.0/Kernel/include
    LIBS += -L$$(HOME)/SDK/RobotSDK_4.0/Kernel/lib -lKernel_Debug
    include($$(HOME)/SDK/RobotSDK_4.0/RobotSDK.pri)
}

win32 {
    INCLUDEPATH += c:/SDK/RobotSDK_4.0/Kernel/include
    LIBS += -Lc:/SDK/RobotSDK_4.0/Kernel/lib -lKernel_Debug
    include(c:/SDK/RobotSDK_4.0/RobotSDK.pri)
}

SOURCES += \
    numberviewer.cpp \
    randomgenerator.cpp

HEADERS += \
    numberviewer.h \
    randomgenerator.h

unix {
    target.path = /usr/lib
    INSTALLS += target
}

