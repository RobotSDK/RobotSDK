#include "xedge.h"

XEdge::XEdge(QGraphicsItem *parent)
    : QGraphicsPathItem(parent)
{
    setCursor(Qt::OpenHandCursor);
    setPen(QPen(QColor(Qt::black)));
}

XEdge::~XEdge()
{

}

void XEdge::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    if(event->button()==Qt::RightButton)
    {
        emit signalRemoveEdge(outputnodefullname,outputportid,inputnodefullname,inputportid);
    }
}
