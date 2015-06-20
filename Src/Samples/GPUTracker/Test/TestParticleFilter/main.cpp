#include <QCoreApplication>

#include<iostream>

extern "C" void cudaMain();

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    cudaMain();

    return a.exec();
}
