#-------------------------------------------------
#
# Project created by QtCreator 2015-04-27T18:19:52
#
#-------------------------------------------------

QT       -= gui

TARGET = LIDAR
TEMPLATE = lib

SOURCES += \
    LIDARSensor.cpp

HEADERS += \
    LIDARSensor.h

include($$(HOME)/SDK/RobotSDK_4.0/Kernel/RobotSDK.pri)

unix{
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_core
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_highgui
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_features2d
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_objdetect
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_contrib
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_imgproc
}
