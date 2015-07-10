#include "xrobot.h"
#include "ui_xrobot.h"

using namespace RobotX;

XConfig * xconfig;

XRobot::XRobot(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::XRobot)
{
    ui->setupUi(this);
    view=new GraphView;
    graph=new XGraph;
    view->setScene(graph);
    ui->layout->addWidget(view);
    connect(view,SIGNAL(signalHandleMenu()), graph, SLOT(slotHandleMenu()));

    xconfig=new XConfig;
    this->addDockWidget(Qt::RightDockWidgetArea,xconfig->configpanel);
//    this->addDockWidget(Qt::LeftDockWidgetArea,xconfig->mainwindowdock);

    QStringList arguments=QApplication::instance()->arguments();
    QString NodeName;
    if(arguments.size()>1)
    {
        NodeName=arguments[1];
    }
    else
    {
        NodeName=QFileInfo(arguments[0]).baseName();
    }
    NodeName.replace(QRegExp("[^a-zA-Z0-9/_$]"),QString("_"));
    this->setWindowTitle(QString("Robot-X : %1").arg(NodeName));
}

XRobot::~XRobot()
{
    delete xconfig;
    delete ui;
}

void XRobot::closeEvent(QCloseEvent *event)
{
    graph->slotCloseGraph();
    QMainWindow::closeEvent(event);
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
            ratio=1.1;
        }
        else
        {
            ratio=0.9;
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
