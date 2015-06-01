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
include($$(ROBOTSDKMODULE))

unix{
    INCLUDEPATH += /usr/include
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_core
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_highgui
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_features2d
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_objdetect
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_contrib
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_imgproc

    INCLUDEPATH += $$(HOME)/Git/Autoware/ros/devel/include
}
