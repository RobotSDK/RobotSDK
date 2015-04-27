#-------------------------------------------------
#
# Project created by QtCreator 2015-04-27T12:04:38
#
#-------------------------------------------------

QT       -= gui

TARGET = Camera
TEMPLATE = lib

SOURCES += \
    CameraSensor.cpp \
    ImageViewer.cpp \
    ImageProcessor.cpp

HEADERS += \
    CameraSensor.h \
    ImageViewer.h \
    ImageProcessor.h

include($$(HOME)/SDK/RobotSDK_4.0/Kernel/RobotSDK.pri)

unix{
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_core
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_highgui
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_features2d
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_objdetect
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_contrib
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_imgproc
}
