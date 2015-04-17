QT += widgets xml opengl
TEMPLATE = lib
CONFIG += staticlib qt
CONFIG += c++11

HEADERS += \
   defines.h \
    graph.h \
    node.h \
    port.h \
    valuebase.h \
    xmldominterface.h \
    sync.h

SOURCES += \
    graph.cpp \
    node.cpp \
    port.cpp \
    valuebase.cpp \
    xmldominterface.cpp \
    sync.cpp

OTHER_FILES += \
    RobotSDK.h

DISTFILES += \
    RobotSDK.pri

INCLUDEPATH += .

unix{
    ROBOTSDKDIR=$$(HOME)/SDK/RobotSDK_4.0

    HEADERS += \
        glviewer.h

    SOURCES += \
        glviewer.cpp

    INCLUDEPATH += /usr/include/eigen3

    TMPROS=indigo

    ROS = $$(ROS_DISTRO)
    isEmpty(ROS){
        isEmpty(TMPROS){
        }
        else
        {
            HEADERS += \
               rosinterface.h
            SOURCES += \
               rosinterface.cpp
            INCLUDEPATH += /opt/ros/$$TMPROS/include
        }
    }
    else{
        HEADERS += \
           rosinterface.h
        SOURCES += \
           rosinterface.cpp
        INCLUDEPATH += /opt/ros/$$(ROS_DISTRO)/include
    }
}

win32{
    ROBOTSDKDIR=C:/SDK/RobotSDK_4.0

    EIGEN=$$(EIGEN_PATH)
    isEmpty(EIGEN){
    }
    else{
        HEADERS += \
            glviewer.h

        SOURCES += \
            glviewer.cpp

        INCLUDEPATH += $$(EIGEN_PATH)
    }
}

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

robotsdkhelper.files = $$DISTFILES
robotsdkhelper.path = $$ROBOTSDKDIR

INSTALLS += robotsdkhelper



