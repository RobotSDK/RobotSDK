#include "scaninterfacefunction.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    ScanInterfaceFunction w;
    w.show();

    return a.exec();
}
