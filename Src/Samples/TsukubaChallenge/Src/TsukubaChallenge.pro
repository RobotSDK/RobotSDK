#-------------------------------------------------
#
# Project created by QtCreator 2015-05-14T12:41:53
#
#-------------------------------------------------

QT       -= gui

TARGET = TsukubaChallenge
TEMPLATE = lib

DEFINES += TSUKUBACHALLENGE_LIBRARY

SOURCES += \
    Controller.cpp \
    Localization.cpp \
    Obstacle.cpp \
    Planning.cpp \
    PlanningSim.cpp

HEADERS += \
    Controller.h \
    Localization.h \
    Obstacle.h \
    Planning.h \
    PlanningSim.h

include($$(HOME)/SDK/RobotSDK_4.0/Kernel/RobotSDK.pri)

INCLUDEPATH += /usr/local/include
LIBS += -L/usr/local/lib -lSHSpur
