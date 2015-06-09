#include "DPMModifierWidget.h"

DPMRect::DPMRect(QString rectCategory, uint rectID, QColor rectColor, qreal x, qreal y, qreal width, qreal height, QGraphicsItem *parent)
{
    category=rectCategory;
    id=rectID;
    color=rectColor;
    this->setToolTip(QString("%1_%2").arg(category).arg(id));
    this->setZValue(1);
    this->setPen(QPen(color));
    this->setBrush(Qt::NoBrush);
    this->setFlag(QGraphicsItem::ItemIsMovable);
    this->setAcceptHoverEvents(1);
}

void DPMRect::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    if(event->button()==Qt::LeftButton)
    {
        this->setZValue(2);
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
    this->setZValue(1);
    QGraphicsRectItem::mouseReleaseEvent(event);
}

void DPMRect::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
    this->setPen(QPen(Qt::red));
    this->update();
    this->setCursor(Qt::OpenHandCursor);
}

void DPMRect::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
    this->setPen(QPen(color));
    this->update();
    this->setCursor(Qt::ArrowCursor);
}

void DPMRect::hoverMoveEvent(QGraphicsSceneHoverEvent *event)
{
    this->setPen(QPen(Qt::red));
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

DPMModifierWidget::DPMModifierWidget(QStringList categories, QWidget *parent)
    : QGraphicsView(parent)
{
    pixmap=NULL;
    scene=new QGraphicsScene;
    this->setScene(scene);
    uint i,n=categories.size();
    cv::Mat graymap(n,1,CV_8UC1);
    for(i=0;i<=n;i++)
    {
        graymap.at<uchar>(i)=i;
    }
    cv::Mat colormap;
    cv::applyColorMap(graymap,colormap,cv::COLORMAP_RAINBOW);
    for(i=0;i<n;i++)
    {
        QString category=categories.at(i);
        cv::Vec3b color=colormap.at<cv::Vec3b>(i);
        colortable.insert(category,QColor(color[0],color[1],color[2]));
        filter.insert(category,1);
        idcount.insert(category,0);
    }
}

void DPMModifierWidget::slotDeleteRect(QGraphicsRectItem *rect)
{
    scene->removeItem(rect);
    delete rect;
}

void DPMModifierWidget::slotMoveForward(QGraphicsRectItem *rect)
{
    QList<QGraphicsItem *> items=scene->items();
    if(items.front()!=rect)
    {
        rect->stackBefore(items.front());
    }
}

void DPMModifierWidget::slotMoveBackward(QGraphicsRectItem *rect)
{
    QList<QGraphicsItem *> items=scene->items();
    uint i,n=items.size();
    for(i=0;i<n;i++)
    {
        if(items.at(i)!=rect)
        {
            break;
        }
    }
    for(i=i+1;i<n;i++)
    {
        if(items.at(i)!=pixmap)
        {
            items.at(i)->stackBefore(rect);
        }
    }
}

void DPMModifierWidget::addPixmap(QImage &image)
{
    pixmap=new QGraphicsPixmapItem(QPixmap::fromImage(image));
    pixmap->setPos(0,0);
    pixmap->setZValue(0);
    scene->addItem(pixmap);
}

void DPMModifierWidget::addRect(QString rectCategory, uint rectID, qreal x, qreal y, qreal width, qreal height)
{
    if(colortable.contains(rectCategory)&&filter.value(rectCategory))
    {
        DPMRect * rect=new DPMRect(rectCategory,rectID,colortable.value(rectCategory),x,y,width,height);
        scene->addItem(rect);
    }
}

QVector<DPMRect *> DPMModifierWidget::getRects()
{
    QList<QGraphicsItem *> rects=scene->items();
    QVector<DPMRect *> result;
    uint i,n=rects.size();
    for(i=0;i<n;i++)
    {
        if(rects[i]!=pixmap)
        {
            result.push_back((DPMRect *)rect);
        }
    }
    return result;
}

void DPMModifierWidget::mousePressEvent(QMouseEvent *event)
{
    if(event->button()==Qt::RightButton)
    {
        QGraphicsItem * item=this->itemAt(event->pos());
        if(item==NULL||item==pixmap)
        {
            QMenu menu;
            QMap<QString, bool>::const_iterator iter;
            for(iter=filter.begin();iter!=filter.end();iter++)
            {
                if(iter.value())
                {
                    menu.addAction(QString("Add %1").arg(iter.key()));
                }
            }
            QAction * selitem=menu.exec(QCursor::pos());
            if(selitem)
            {
                QString category=selitem->text();
                QColor=colortable.value(category);
                QPointF pos=mapToScene(event->pos());
                this->addRect(category,pos.x(),pos.y(),100,100);
            }
            return;
        }
    }
}
