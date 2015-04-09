QT += widgets xml
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

unix{
    ROBOTSDKDIR=$$(HOME)/SDK/RobotSDK_4.0

    MOC_DIR = $$ROBOTSDKDIR/Build/Kernel/MOC
    UI_DIR = $$ROBOTSDKDIR/Build/Kernel/UI

    CONFIG(debug, debug|release){
        OBJECTS_DIR = $$ROBOTSDKDIR/Build/Kernel/OBJ/Debug
        DESTDIR = $$ROBOTSDKDIR/Build/Kernel/lib
        TARGET = Kernel_Debug
        target.path = $$ROBOTSDKDIR/Kernel/lib
    }
    else {
        OBJECTS_DIR = $$ROBOTSDKDIR/Build/Kernel/OBJ/Release
        DESTDIR = $$ROBOTSDKDIR/Build/Kernel/lib
        TARGET = Kernel_Release
        target.path = $$ROBOTSDKDIR/Kernel/lib
    }

    INSTALLS += target

    headertarget.files = $$HEADERS
    headertarget.path = $$ROBOTSDKDIR/Kernel/include

    INSTALLS += headertarget

    robotsdkheader.files = $$OTHER_FILES
    robotsdkheader.path = $$ROBOTSDKDIR/Kernel/include

    INSTALLS += robotsdkheader
}

