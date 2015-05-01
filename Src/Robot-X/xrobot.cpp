#include "xrobot.h"
#include "ui_xrobot.h"

using namespace RobotX;

XRobot::XRobot(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::XRobot)
{
    ui->setupUi(this);

    graph=new XGraph;
    view=new GraphView;
    view->setScene(graph);
    ui->layout->addWidget(view);
    connect(view,SIGNAL(signalHandleMenu()), graph, SLOT(slotHandleMenu()));
}

XRobot::~XRobot()
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
    if(ctrlflag)
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
    else
    {
        QGraphicsView::wheelEvent(event);
    }
}

void GraphView::keyPressEvent(QKeyEvent *event)
{
    switch(event->key())
    {
    case Qt::Key_Control:
        ctrlflag=1;
        break;
    default:
        break;
    }
    QGraphicsView::keyPressEvent(event);
}

void GraphView::keyReleaseEvent(QKeyEvent *event)
{
    switch(event->key())
    {
    case Qt::Key_Control:
        ctrlflag=0;
        break;
    default:
        break;
    }
    QGraphicsView::keyReleaseEvent(event);
}
