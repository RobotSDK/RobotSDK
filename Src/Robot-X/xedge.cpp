#include "xedge.h"

XEdge::XEdge(QGraphicsItem *parent)
    : QGraphicsPathItem(parent)
{
    setCursor(Qt::OpenHandCursor);
    setPen(QPen(QColor(Qt::black)));
    setZValue(-1);
}

XEdge::~XEdge()
{

}

void XEdge::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    setCursor(Qt::ClosedHandCursor);
    this->setPen(QColor(Qt::red));
    this->setZValue(1);
    this->update();

    if(event->button()==Qt::RightButton)
    {
        QMenu menu;
        menu.addAction("Delete Edge");
        QAction * selecteditem=menu.exec(QCursor::pos());
        if(selecteditem)
        {
            if(selecteditem->text()==QString("Delete Edge"))
            {
                emit signalRemoveEdge(outputnodefullname,outputportid,inputnodefullname,inputportid);
            }
        }
    }
}

void XEdge::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    Q_UNUSED(event);
    this->setCursor(Qt::OpenHandCursor);
    this->setPen(QColor(Qt::black));
    this->setZValue(-1);
    this->update();
}
