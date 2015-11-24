#-------------------------------------------------
#
# Project created by QtCreator 2015-04-27T12:16:48
#
#-------------------------------------------------

QT       -= gui

TARGET = Localization
TEMPLATE = lib

SOURCES += \
    NDTLocalizer.cpp \
    PathViewer.cpp

HEADERS += \
    NDTLocalizer.h \
    PathViewer.h

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
