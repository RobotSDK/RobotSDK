#include "DPMModifierWidgets.h"

DPMViewer::DPMViewer(QWidget *parent)
    : QGraphicsView(parent)
{
    pixmap=NULL;
    scene=new QGraphicsScene;
    this->setScene(scene);
}

void DPMViewer::slotDeleteRect(QGraphicsRectItem *rect)
{
    scene->removeItem(rect);
    delete rect;
}

void DPMViewer::clear()
{
    scene->clear();
}

void DPMViewer::addPixmap(QImage &image)
{
    pixmap=new QGraphicsPixmapItem(QPixmap::fromImage(image));
    pixmap->setPos(0,0);
    pixmap->setZValue(-1);
    scene->addItem(pixmap);
}

void DPMViewer::addRect(qreal x, qreal y, qreal width, qreal height)
{
    DPMRect * rect=new DPMRect(x,y,width,height);
    scene->addItem(rect);
}

QVector<QRectF> DPMViewer::getRects()
{
    QList<QGraphicsItem *> rects=this->items();
    QVector<QRectF> result;
    uint i,n=rects.size();
    for(i=0;i<n;i++)
    {
        if(rects[i]!=pixmap)
        {
            QGraphicsRectItem * rect=static_cast<QGraphicsRectItem *>(rects[i]);
            result.push_back(rect->rect());
        }
    }
    return result;
}

void DPMViewer::mousePressEvent(QMouseEvent *event)
{
    if(event->button()==Qt::RightButton)
    {
        QGraphicsItem * item=this->itemAt(event->pos());
        if(item==NULL||item==pixmap)
        {
            QMenu menu;
            menu.addAction("Add Rectangle");
            QAction * selitem=menu.exec(QCursor::pos());
            if(selitem)
            {
                if(selitem->text()=="Add Rectangle")
                {
                    QPointF pos=mapToScene(event->pos());
                    this->addRect(pos.x(),pos.y(),100,100);
                }
            }
            return;
        }
        else if(item!=NULL)
        {
            QGraphicsRectItem * rect=static_cast<QGraphicsRectItem *>(item);
            rect->setPen(QPen(Qt::green));
            rect->update();
            QMenu menu;
            menu.addAction("Delete Rectangle");
            menu.addSeparator();
            menu.addAction("Top Level");
            menu.addAction("Bottom Level");
            QAction * selitem=menu.exec(QCursor::pos());
            if(selitem)
            {
                if(selitem->text()=="Delete Rectangle")
                {
                    slotDeleteRect(rect);
                }
                else if(selitem->text()=="Top Level")
                {
                    rect->setZValue(2);
                    rect->setPen(QPen(Qt::red));
                    rect->update();
                }else if(selitem->text()=="Bottom Level")
                {
                    rect->setZValue(0);
                    rect->setPen(QPen(Qt::red));
                    rect->update();
                }
            }
        }
    }
    QGraphicsView::mousePressEvent(event);
}

DPMRect::DPMRect(qreal x, qreal y, qreal width, qreal height, QGraphicsItem *parent)
    : QGraphicsRectItem(x,y,width,height,parent)
{
    this->setZValue(1);
    this->setPen(QPen(Qt::red));
    this->setBrush(Qt::NoBrush);
    this->setFlag(QGraphicsItem::ItemIsMovable);
//    this->setFlag(QGraphicsItem::ItemIsSelectable);
    this->setAcceptHoverEvents(1);
}

void DPMRect::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    if(event->button()==Qt::LeftButton)
    {
        oriz=this->zValue();
        this->setZValue(3);
        if(edgeflag||cornerflag)
        {
            orirect=this->rect();
            resizeflag=1;
        }
        else
        {
            resizeflag=0;
            QGraphicsRectItem::mousePressEvent(event);
        }
    }
    else
    {
        QGraphicsRectItem::mousePressEvent(event);
    }
}

void DPMRect::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    if(resizeflag)
    {
        QRectF rect=orirect;
        if(edgeflag)
        {
            switch(edge)
            {
            case Qt::LeftEdge:
                {
                    rect.setX(event->pos().x());
                    break;
                }
            case Qt::RightEdge:
                {
                    rect.setWidth(event->pos().x()-rect.x());
                    break;
                }
            case Qt::TopEdge:
                {
                    rect.setY(event->pos().y());
                    break;
                }
            case Qt::BottomEdge:
                {
                    rect.setHeight(event->pos().y()-rect.y());
                    break;
                }
            }
        }
        else
        {
            switch(corner)
            {
            case Qt::TopLeftCorner:
                {
                    rect.setX(event->pos().x());
                    rect.setY(event->pos().y());
                }
                break;
            case Qt::TopRightCorner:
                {
                    rect.setWidth(event->pos().x()-rect.x());
                    rect.setY(event->pos().y());
                }
                break;
            case Qt::BottomLeftCorner:
                {
                    rect.setX(event->pos().x());
                    rect.setHeight(event->pos().y()-rect.y());
                }
                break;
            case Qt::BottomRightCorner:
                {
                    rect.setWidth(event->pos().x()-rect.x());
                    rect.setHeight(event->pos().y()-rect.y());
                }
                break;
            }
        }
        this->setRect(rect);
        this->prepareGeometryChange();
    }
    else
    {
        QGraphicsRectItem::mouseMoveEvent(event);
    }
}

void DPMRect::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    resizeflag=0;
    this->setZValue(oriz);
    QGraphicsRectItem::mouseReleaseEvent(event);
}

void DPMRect::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
    this->setPen(QPen(Qt::green));
    this->update();
    this->setCursor(Qt::OpenHandCursor);
}

void DPMRect::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
    this->setPen(QPen(Qt::red));
    this->update();
    this->setCursor(Qt::ArrowCursor);
}

void DPMRect::hoverMoveEvent(QGraphicsSceneHoverEvent *event)
{
    this->setPen(QPen(Qt::green));
    this->update();
    qreal width=this->rect().x()+this->rect().width();
    qreal height=this->rect().y()+this->rect().height();
    qreal boundary=5;
    if(event->pos().x()<=this->rect().x()+boundary)
    {
        if(event->pos().y()<=this->rect().y()+boundary)
        {
            edgeflag=0;
            cornerflag=1;
            corner=Qt::TopLeftCorner;
            this->setCursor(Qt::SizeFDiagCursor);
        }
        else if(event->pos().y()>=height-1-boundary)
        {
            edgeflag=0;
            cornerflag=1;
            corner=Qt::BottomLeftCorner;
            this->setCursor(Qt::SizeBDiagCursor);
        }
        else
        {
            edgeflag=1;
            cornerflag=0;
            edge=Qt::LeftEdge;
            this->setCursor(Qt::SizeHorCursor);
        }
    }
    else if(event->pos().x()>=width-1-boundary)
    {
        if(event->pos().y()<=this->rect().y()+boundary)
        {
            edgeflag=0;
            cornerflag=1;
            corner=Qt::TopRightCorner;
            this->setCursor(Qt::SizeBDiagCursor);
        }
        else if(event->pos().y()>=height-1-boundary)
        {
            edgeflag=0;
            cornerflag=1;
            corner=Qt::BottomRightCorner;
            this->setCursor(Qt::SizeFDiagCursor);
        }
        else
        {
            edgeflag=1;
            cornerflag=0;
            edge=Qt::RightEdge;
            this->setCursor(Qt::SizeHorCursor);
        }
    }
    else
    {
        if(event->pos().y()<=this->rect().y()+boundary)
        {
            edgeflag=1;
            cornerflag=0;
            edge=Qt::TopEdge;
            this->setCursor(Qt::SizeVerCursor);
        }
        else if(event->pos().y()>=height-1-boundary)
        {
            edgeflag=1;
            cornerflag=0;
            edge=Qt::BottomEdge;
            this->setCursor(Qt::SizeVerCursor);
        }
        else
        {
            edgeflag=0;
            cornerflag=0;
            this->setCursor(Qt::OpenHandCursor);
        }
    }
}
