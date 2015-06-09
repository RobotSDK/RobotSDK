#ifndef XCONFIG_H
#define XCONFIG_H

#include <QDockWidget>
#include <QTabWidget>
#include <QMdiArea>

namespace RobotX
{

class XConfig
{
public:
    XConfig();
public:
    QDockWidget * configpanel;
    QTabWidget * configtab;
//    QDockWidget * mainwindowdock;
//    QMdiArea * mainwindow;
};

}

#endif // XCONFIG_H
