#-------------------------------------------------
#
# Project created by QtCreator 2015-04-27T17:59:31
#
#-------------------------------------------------

QT       -= gui

TARGET = DPM
TEMPLATE = lib

DEFINES += DPM_LIBRARY

SOURCES += \
    DPMDetector.cpp \
    DPMModifier.cpp \
    DPMModifierWidgets.cpp

HEADERS += \
    DPMDetector.h \
    DPMModifier.h \
    DPMModifierWidgets.h

MODULES += Camera
include($$(HOME)/SDK/RobotSDK_4.0/Kernel/RobotSDK.pri)

unix{
    INCLUDEPATH += /usr/local/include
    LIBS += -L/usr/local/lib -lopencv_core
    LIBS += -L/usr/local/lib -lopencv_highgui
    LIBS += -L/usr/local/lib -lopencv_features2d
    LIBS += -L/usr/local/lib -lopencv_objdetect
    LIBS += -L/usr/local/lib -lopencv_contrib
    LIBS += -L/usr/local/lib -lopencv_imgproc

    INCLUDEPATH += $$(HOME)/Git/Autoware/ros/devel/include
}
