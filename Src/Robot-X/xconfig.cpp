#include "xconfig.h"

using namespace RobotX;

XConfig::XConfig()
{
    configpanel=new QDockWidget("Config Panel");
    configtab=new QTabWidget;
    configpanel->setWidget(configtab);
    configpanel->setVisible(0);
    configpanel->setFeatures(QDockWidget::DockWidgetClosable);
    configtab->setTabPosition(QTabWidget::West);

//    mainwindowdock=new QDockWidget("Widgets Window");
//    mainwindow=new QMdiArea;
//    mainwindowdock->setWidget(mainwindow);
//    mainwindowdock->setVisible(0);
}
