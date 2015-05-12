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
