#-------------------------------------------------
#
# Project created by QtCreator 2014-08-12T01:53:41
#
#-------------------------------------------------

QT += widgets network opengl xml

CONFIG += staticlib qt
TEMPLATE = lib
TARGET = Kernel

SOURCES += \
    Core/Node/node.cpp \
    Core/Edge/edge.cpp \
    Core/Edge/triggerlog.cpp \
    Core/Edge/triggerview.cpp \
    Modules/Drain/drain.cpp \
    Modules/Processor/processor.cpp \
    Modules/Source/source.cpp \
    Modules/SourceDrain/sourcedrain.cpp \
    ExModules/Drain/Storage/storage.cpp \
    ExModules/Drain/Transmitter/transmitter.cpp \
    ExModules/Drain/Visualization/visualization.cpp \
    ExModules/Source/Sensor/sensor.cpp \
    ExModules/Source/Simulator/simulator.cpp \
    ExModules/Source/UserInput/userinput.cpp \
    Accessories/XMLDomInterface/xmldominterface.cpp

HEADERS += \
    Core/Node/node.h \
    Core/Edge/edge.h \
    Core/Edge/triggerlog.h \
    Core/Edge/triggerview.h \
    Modules/Drain/drain.h \
    Modules/Processor/processor.h \
    Modules/Source/source.h \
    Modules/SourceDrain/sourcedrain.h \
    ExModules/Drain/Storage/storage.h \
    ExModules/Drain/Transmitter/transmitter.h \
    ExModules/Drain/Visualization/visualization.h \
    ExModules/Source/Sensor/sensor.h \
    ExModules/Source/Simulator/simulator.h \
    ExModules/Source/UserInput/userinput.h \
    Accessories/XMLDomInterface/xmldominterface.h \
    RobotSDK_Global.h

OTHER_FILES += \
    CreateRule.xml \
    Interface_Functions.xml

INCLUDEPATH += .

unix{
    INCLUDEPATH += /usr/include
	
	CONFIG(debug, debug|release){
		DESTDIR = $$(HOME)/Build/RobotSDK/Kernel/lib/Debug
		target.path = $$(HOME)/SDK/RobotSDK/Kernel/lib/Debug
	}
	else {
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
	
	CONFIG(debug, debug|release){
		DESTDIR = $$(RobotSDK_Kernel)/../../../Build/RobotSDK/Kernel/lib/Debug
		target.path = $$(RobotSDK_Kernel)/lib/Debug
	}
	else {
		DESTDIR = $$(RobotSDK_Kernel)/../../../Build/RobotSDK/Kernel/lib/Release
		target.path = $$(RobotSDK_Kernel)/lib/Release
	}
    
    INSTALLS += target

    KERNELPATH = $$(RobotSDK_Kernel)/include
	
	INSTALL_PREFIX = $$KERNELPATH
    INSTALL_HEADERS = $$HEADERS
    include(Kernel.pri)

	
	include(Kernel.pri)

	robotsdk_createrule.path = $$KERNELPATH
    robotsdk_createrule.files = $$OTHER_FILES
    INSTALLS += robotsdk_createrule
}
