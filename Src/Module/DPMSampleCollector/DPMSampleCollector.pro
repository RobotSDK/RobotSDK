#-------------------------------------------------
#
# Project created by QtCreator 2015-04-22T19:37:12
#
#-------------------------------------------------

QT       -= gui

TARGET = DPMSampleCollector
TEMPLATE = lib

DEFINES += DPMSAMPLECOLLECTOR_LIBRARY

SOURCES += \
    CameraSensor.cpp \
    ImageProcessor.cpp \
    ImageViewer.cpp \
    DPMDetector.cpp \
    DPMModifier.cpp \
    DPMModifierWidgets.cpp

HEADERS += \
    CameraSensor.h \
    ImageProcessor.h \
    ImageViewer.h \
    DPMDetector.h \
    DPMModifier.h \
    DPMModifierWidgets.h

unix {
    target.path = /usr/lib
    INSTALLS += target
}

include($$(HOME)/SDK/RobotSDK_4.0/Kernel/RobotSDK.pri)

unix{
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_core
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_highgui
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_features2d
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_objdetect
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_contrib
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_imgproc

    INCLUDEPATH += $$(HOME)/Git/Autoware/ros/devel/include
}
