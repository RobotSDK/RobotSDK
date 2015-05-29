#include "xrobot.h"
#ifdef Q_OS_LINUX
#include "rosinterface.h"
#endif
#include <QApplication>

using namespace RobotX;

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

#ifdef Q_OS_LINUX
    if(argc>1)
    {
        ROSInterfaceBase::NodeName=QString(argv[1]);
    }
#endif
    XRobot w;
    w.show();

    return a.exec();
}
