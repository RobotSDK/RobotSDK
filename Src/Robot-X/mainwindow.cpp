#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    graph=new XGraph;
    ui->graphicsView->setScene(graph);

    QString libraryname=QString("/home/alexanderhmw/Git/dev/RobotSDK/Src/Sample/build-TestModule-Desktop_Qt_5_4_1_GCC_64bit-Debug/libTestModule.so");

    graph->slotAddNode("RandomGenerator::random",libraryname);
    graph->slotAddNode("NumberViewer::odd",libraryname);
    graph->slotAddNode("NumberViewer::even",libraryname);
}

MainWindow::~MainWindow()
{
    delete ui;
}
