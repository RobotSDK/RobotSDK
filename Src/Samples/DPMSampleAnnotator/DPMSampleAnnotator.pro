#-------------------------------------------------
#
# Project created by QtCreator 2015-06-08T15:26:53
#
#-------------------------------------------------

QT       -= gui

TARGET = DPMSampleAnnotator
TEMPLATE = lib

DEFINES += DPMSAMPLEANNOTATOR_LIBRARY

SOURCES += \
    DPMModifier.cpp \
    DPMSampleLoader.cpp \
    DPMSampleSaver.cpp \
    DPMReceiver.cpp \
    ROSBagLoader.cpp \
    DPMAnnotator.cpp \
    DPMModifierWidget.cpp \
    DPMAnnotatorWidget.cpp

HEADERS += \
    DPMAnnotator.h \
    DPMModifier.h \
    DPMSampleLoader.h \
    DPMSampleSaver.h \
    DPMReceiver.h \
    ROSBagLoader.h \
    DPMModifierWidget.h \
    DPMAnnotatorWidget.h

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
