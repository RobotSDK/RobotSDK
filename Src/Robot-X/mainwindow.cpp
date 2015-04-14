#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    graph=new XGraph;
    view=new GraphView;
    view->setScene(graph);
    ui->layout->addWidget(view);
    connect(view,SIGNAL(signalHandleMenu()), graph, SLOT(slotHandleMenu()));
}

MainWindow::~MainWindow()
{
    delete ui;
}

GraphView::GraphView(QWidget *parent)
    : QGraphicsView(parent)
{

}

void GraphView::mousePressEvent(QMouseEvent *event)
{
    if(event->button()==Qt::RightButton)
    {
        QGraphicsItem * item=itemAt(event->pos());
        if(item==NULL)
        {
            emit signalHandleMenu();
        }
    }
    QGraphicsView::mousePressEvent(event);
}

void GraphView::wheelEvent(QWheelEvent *event)
{
    if(event->delta()>0)
    {
        ratio=0.9;
    }
    else
    {
        ratio=1.1;
    }
    this->scale(ratio,ratio);
}
