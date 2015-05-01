#-------------------------------------------------
#
# Project created by QtCreator 2015-04-11T17:52:19
#
#-------------------------------------------------

QT       += core gui xml

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Robot-X
TEMPLATE = app

CONFIG += c++11

unix {
    INCLUDEPATH += $$(HOME)/SDK/RobotSDK_4.0/Kernel/include
    CONFIG(debug, debug|release){
        LIBS += -L$$(HOME)/SDK/RobotSDK_4.0/Kernel/lib -lKernel_Debug
    }
    else{
        LIBS += -L$$(HOME)/SDK/RobotSDK_4.0/Kernel/lib -lKernel_Release
    }

    INCLUDEPATH += /usr/include/graphviz
    LIBS += -L/usr/lib -lcgraph
    LIBS += -L/usr/lib -lgvc
}

win32 {
    INCLUDEPATH += C:/SDK/RobotSDK_4.0/Kernel/include
    CONFIG(debug, debug|release){
        LIBS += -LC:/SDK/RobotSDK_4.0/Kernel/lib -lKernel_Debug
    }
    else{
        LIBS += -LC:/SDK/RobotSDK_4.0/Kernel/lib -lKernel_Release
    }

    GRAPHVIZ=$$(GRAPHVIZ_PATH)
    isEmpty(GRAPHVIZ){
        error(GRAPHVIZ_PATH is not set)
    }
    else{
        INCLUDEPATH += $$(GRAPHVIZ_PATH)/include/graphviz
        CONFIG(debug, debug|release){
            LIBS += -L$$(GRAPHVIZ_PATH)/lib/debug/lib -lcgraph
            LIBS += -L$$(GRAPHVIZ_PATH)/lib/debug/lib -lgvc
        }
        else{
            LIBS += -L$$(GRAPHVIZ_PATH)/lib/release/lib -lcgraph
            LIBS += -L$$(GRAPHVIZ_PATH)/lib/release/lib -lgvc
        }
    }
}

SOURCES += main.cpp\
    xnode.cpp \
    xedge.cpp \
    xport.cpp \
    xgraph.cpp \
    xrobot.cpp

HEADERS  += \
    xnode.h \
    xedge.h \
    xport.h \
    xgraph.h \
    xrobot.h

FORMS    += \
    xrobot.ui

unix{
    ROBOTSDKDIR=$$(HOME)/SDK/RobotSDK_4.0
}

win32{
    ROBOTSDKDIR=C:/SDK/RobotSDK_4.0
}

MOC_DIR = $$ROBOTSDKDIR/Build/Robot-X/MOC
UI_DIR = $$ROBOTSDKDIR/Build/Robot-X/UI

CONFIG(debug, debug|release){
        OBJECTS_DIR = $$ROBOTSDKDIR/Build/Robot-X/OBJ/Debug
        DESTDIR = $$ROBOTSDKDIR/Build/Robot-X
        TARGET = Robot-X_Debug
        target.path = $$ROBOTSDKDIR/Robot-X
}
else {
        OBJECTS_DIR = $$ROBOTSDKDIR/Build/Robot-X/OBJ/Release
        DESTDIR = $$ROBOTSDKDIR/Build/Robot-X
        TARGET = Robot-X
        target.path = $$ROBOTSDKDIR/Robot-X
}

INSTALLS += target
