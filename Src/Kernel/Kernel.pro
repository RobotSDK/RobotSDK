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
    ../Accessories/XMLDomInterface/xmldominterface.cpp \
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

unix{
    INCLUDEPATH += /usr/include

    MOC_DIR = $$(HOME)/Build/RobotSDK/Kernel/MOC
    UI_DIR = $$(HOME)/Build/RobotSDK/Kernel/UI

    CONFIG(debug, debug|release){
        OBJECTS_DIR = $$(HOME)/Build/RobotSDK/Kernel/OBJ/Debug
        DESTDIR = $$(HOME)/Build/RobotSDK/Kernel/lib/Debug
        target.path = $$(HOME)/SDK/RobotSDK/Kernel/lib/Debug
    }
    else {
        OBJECTS_DIR = $$(HOME)/Build/RobotSDK/Kernel/OBJ/Release
        DESTDIR = $$(HOME)/Build/RobotSDK/Kernel/lib/Release
        target.path = $$(HOME)/SDK/RobotSDK/Kernel/lib/Release
    }

    INSTALLS += target

    KERNELPATH = $$(HOME)/SDK/RobotSDK/Kernel/include

    INSTALL_PREFIX = $$KERNELPATH
    INSTALL_HEADERS = $$HEADERS
    include(Kernel.pri)

    robotsdk_createrule.path = $$KERNELPATH
    robotsdk_createrule.files = $$OTHER_FILES
    INSTALLS += robotsdk_createrule
}

win32{
    TMPPATH=$$(RobotDep_Include)
    isEmpty(TMPPATH) {
        error($$TMPPATH is not Specified.)
    }
    else{
        INCLUDEPATH += $$split(TMPPATH,;)
    }

    MOC_DIR = $$(RobotSDK_Kernel)/../../../Build/RobotSDK/Kernel/MOC

    UI_DIR = $$(RobotSDK_Kernel)/../../../Build/RobotSDK/Kernel/UI

    CONFIG(debug, debug|release){
        OBJECTS_DIR = $$(RobotSDK_Kernel)/../../../Build/RobotSDK/Kernel/OBJ/Debug
        DESTDIR = $$(RobotSDK_Kernel)/../../../Build/RobotSDK/Kernel/lib/Debug
        target.path = $$(RobotSDK_Kernel)/lib/Debug
    }
    else {
        OBJECTS_DIR = $$(RobotSDK_Kernel)/../../../Build/RobotSDK/Kernel/OBJ/Release
        DESTDIR = $$(RobotSDK_Kernel)/../../../Build/RobotSDK/Kernel/lib/Release
        target.path = $$(RobotSDK_Kernel)/lib/Release
    }

    INSTALLS += target

    KERNELPATH = $$(RobotSDK_Kernel)/include

    INSTALL_PREFIX = $$KERNELPATH
    INSTALL_HEADERS = $$HEADERS
    include(Kernel.pri)

    robotsdk_createrule.path = $$KERNELPATH
    robotsdk_createrule.files = $$OTHER_FILES
    INSTALLS += robotsdk_createrule
}
