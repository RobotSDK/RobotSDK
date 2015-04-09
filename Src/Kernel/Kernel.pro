QT += widgets xml
TARGET = Kernel
TEMPLATE = lib
CONFIG += staticlib qt
CONFIG += c++11

INCLUDEPATH += .

DEFINES += RobotSDK_Kernel

HEADERS += \
    defines.h \
    graph.h \
    node.h \
    port.h \
    valuebase.h \
    xmldominterface.h

SOURCES += \
    graph.cpp \
    node.cpp \
    port.cpp \
    valuebase.cpp \
    xmldominterface.cpp

OTHER_FILES += \
    RobotSDK_Global.h
