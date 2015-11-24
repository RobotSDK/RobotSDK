#-------------------------------------------------
#
# Project created by QtCreator 2015-06-23T14:09:15
#
#-------------------------------------------------

QT       -= gui

TARGET = ROSBagLoader
TEMPLATE = lib

SOURCES += \
    ROSBagLoader_Velodyne.cpp

HEADERS += \
    ROSBagLoader_Velodyne.h

include($$(ROBOTSDKMODULE))

unix{
    INCLUDEPATH += /usr/include
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_core
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_highgui
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_features2d
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_objdetect
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_contrib
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_imgproc
}
