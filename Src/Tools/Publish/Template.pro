#======================================================================

QT += core xml widgets gui

CONFIG += qt
TEMPLATE = app

PROJNAME = KITTI_Dataset_Viewer

# SDK, APP, MOD
INSTALLTYPE = APP

unix:INCLUDEPATH += /home/alexanderhmw/SDK/GLViewer/include

unix{
	CONFIG(debug, debug|release){
	   LIBS += -L/home/alexanderhmw/SDK/GLViewer/lib/Debug -lGLViewer
	}
	else {
	   LIBS += -L/home/alexanderhmw/SDK/GLViewer/lib/Release -lGLViewer
	}
}

#======================================================================

include(RobotSDK.pri)
