#-------------------------------------------------
#
# Project created by QtCreator 2015-04-03T19:40:52
#
#-------------------------------------------------

QT       += widgets xml

greaterThan(QT_MAJOR_VERSION, 4): QT += printsupport

TARGET = Kernel
TEMPLATE = lib
CONFIG += staticlib qt
CONFIG += c++11

SOURCES += \
    Core/Port/port.cpp \
    Accessories/XMLDomInterface/xmldominterface.cpp \
    Core/ModuleDev/valuebase.cpp \
    Core/Node/node.cpp

HEADERS += \
    Core/Port/port.h \
    RobotSDK_Global.h \
    Accessories/XMLDomInterface/xmldominterface.h \
    Core/defines.h \
    Core/ModuleDev/defines.h \
    Core/ModuleDev/valuebase.h \
    Core/Node/node.h

INCLUDEPATH += .
