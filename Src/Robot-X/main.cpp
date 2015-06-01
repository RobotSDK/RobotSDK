#include "xrobot.h"

using namespace RobotX;

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    XRobot w;
    w.show();

    return a.exec();
}
