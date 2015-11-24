#include "VehicleDetectorWidget.h"

VehicleDetectorWidget::VehicleDetectorWidget(QWidget * parent)
    : QGraphicsView(parent)
{
    scene=new QGraphicsScene;
    this->setScene(scene);
    startpointflag=0;
}

void VehicleDetectorWidget::mousePressEvent(QMouseEvent *event)
{
    if(event->button()==Qt::LeftButton)
    {
        QGraphicsItem * item=this->itemAt(event->pos());
        if(item==pixmap)
        {
            if(startpointflag)
            {
                startpointflag=0;
                QPointF endpoint=this->mapToScene(event->pos());
                QLineF line(startpoint,endpoint);
                QGraphicsLineItem * lineitem=new QGraphicsLineItem(line,pixmap);
                lineitem->setZValue(1);
                lineitem->setPos(0,0);
                lineitem->setPen(QPen(Qt::black,3));
                lineitem->setFlag(QGraphicsItem::ItemIsMovable);
                lineitem->setFlag(QGraphicsItem::ItemIsSelectable);
            }
            else
            {
                startpointflag=1;
                startpoint=this->mapToScene(event->pos());
            }
        }
        else
        {
            QGraphicsView::mousePressEvent(event);
        }
    }
    else if(event->button()==Qt::RightButton)
    {
        QGraphicsItem * item=this->itemAt(event->pos());
        if(item!=NULL&&item!=pixmap)
        {
            scene->removeItem(item);
            delete item;
        }
        else
        {
            QGraphicsView::mousePressEvent(event);
        }
    }
}

void VehicleDetectorWidget::wheelEvent(QWheelEvent *event)
{
    if(ctrlflag)
    {
        if(event->delta()>0)
        {
            this->scale(1.1,1.1);
        }
        else
        {
            this->scale(0.9,0.9);
        }
    }
    else
    {
        QGraphicsView::wheelEvent(event);
    }
}

void VehicleDetectorWidget::keyPressEvent(QKeyEvent *event)
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

void VehicleDetectorWidget::keyReleaseEvent(QKeyEvent *event)
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

void VehicleDetectorWidget::setPixmap(QImage &image)
{
    scene->clear();
    pixmap=new QGraphicsPixmapItem(QPixmap::fromImage(image));
    pixmap->setZValue(0);
    pixmap->setPos(0,0);
    scene->addItem(pixmap);
}

QVector<QLineF> VehicleDetectorWidget::getDetection()
{
    QVector<QLineF> result;
    QList<QGraphicsItem *> items=scene->items();
    int i,n=items.size();
    for(i=0;i<n;i++)
    {
        if(items[i]!=pixmap)
        {
            QLineF line=((QGraphicsLineItem *)(items[i]))->line();
            QPointF origin=items[i]->pos();
            line.setP1(line.p1()+origin);
            line.setP2(line.p2()+origin);
            result.push_back(line);
        }
    }
    return result;
}

void VehicleDetectorWidget::clear()
{
    scene->clear();
}
