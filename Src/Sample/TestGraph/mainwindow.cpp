#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    QString libraryname=QFileDialog::getOpenFileName(this,"Open TestModule Shared Library");
    QString configfile="config.xml";

    graph.addNode("RandomGenerator::random",libraryname,configfile);
    graph.addNode("NumberViewer::odd",libraryname,configfile);
    graph.addNode("NumberViewer::even",libraryname,configfile);

    graph.addEdge("RandomGenerator::random",0,"NumberViewer::odd",0);
    graph.addEdge("RandomGenerator::random",1,"NumberViewer::even",0);

    ui->graph->setWidget(graph.switcherpanel);

}

MainWindow::~MainWindow()
{
    delete ui;
}
