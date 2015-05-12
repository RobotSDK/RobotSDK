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
    INCLUDEPATH += /usr/local/include
    LIBS += -L/usr/local/lib -lopencv_core
    LIBS += -L/usr/local/lib -lopencv_highgui
    LIBS += -L/usr/local/lib -lopencv_features2d
    LIBS += -L/usr/local/lib -lopencv_objdetect
    LIBS += -L/usr/local/lib -lopencv_contrib
    LIBS += -L/usr/local/lib -lopencv_imgproc
}
