#include "DPMModifierWidget.h"

using namespace RobotSDK_Module;

DPMRect::DPMRect(QString rectCategory, int rectID, QColor rectColor, qreal x, qreal y, qreal width, qreal height, QGraphicsItem *parent)
    : QGraphicsRectItem(x,y,width,height,parent)
{
    category=rectCategory;
    id=rectID;
    color=rectColor;
    this->setToolTip(QString("%1_%2").arg(category).arg(id));
    this->setZValue(1);
    this->setPen(QPen(color, 3));
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
    this->setPen(QPen(Qt::red, 3));
    this->update();
    this->setCursor(Qt::OpenHandCursor);
}

void DPMRect::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
    this->setPen(QPen(color, 3));
    this->update();
    this->setCursor(Qt::ArrowCursor);
}

void DPMRect::hoverMoveEvent(QGraphicsSceneHoverEvent *event)
{
    this->setPen(QPen(Qt::red, 3));
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

DPMModifierWidget::DPMModifierWidget(QWidget *parent)
    : QGraphicsView(parent)
{
    pixmap=NULL;
    scene=new QGraphicsScene;
    this->setScene(scene);
}

void DPMModifierWidget::setCategories(QStringList categories)
{
    int i,n=categories.size();
    int colornum=230;
    cv::Mat graymap(1,colornum,CV_8UC1);
    for(i=0;i<colornum;i++)
    {
        graymap.at<uchar>(i)=i;
    }
    cv::Mat colormap;
    cv::applyColorMap(graymap,colormap,cv::COLORMAP_RAINBOW);
    colortable.clear();
    filter.clear();
    idcount.clear();
    for(i=0;i<n;i++)
    {
        QString category=categories.at(i);
        int colorid=int(i*double(colornum-1)/n);
        cv::Vec3b color=colormap.at<cv::Vec3b>(colorid);
        colortable.insert(category,QColor(color.val[0],color.val[1],color.val[2],128));
        filter.insert(category,1);
        idcount.insert(category,-1);
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
    int i,n=items.size();
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

void DPMModifierWidget::slotSetFilter(QMap<QString, bool> categoryfilter)
{
    filter=categoryfilter;
    QList<QGraphicsItem *> rects=scene->items();
    int i,n=rects.size();
    for(i=0;i<n;i++)
    {
        if(rects[i]!=pixmap)
        {
            DPMRect * rect=(DPMRect *)(rects[i]);
            rect->setVisible(filter.value(rect->category));
        }
    }
}

void DPMModifierWidget::clear()
{
    scene->clear();
}

void DPMModifierWidget::addPixmap(QImage &image)
{
    pixmap=new QGraphicsPixmapItem(QPixmap::fromImage(image));
    pixmap->setPos(0,0);
    pixmap->setZValue(0);
    scene->addItem(pixmap);
    scene->setSceneRect(image.rect());
}

void DPMModifierWidget::addRect(QString rectCategory, int rectID, qreal x, qreal y, qreal width, qreal height)
{
    if(colortable.contains(rectCategory)&&filter.value(rectCategory))
    {
        DPMRect * rect=new DPMRect(rectCategory,rectID,colortable.value(rectCategory),x,y,width,height,pixmap);
        //scene->addItem(rect);
    }
}

QVector<DPMRect *> DPMModifierWidget::getRects()
{
    QList<QGraphicsItem *> rects=scene->items();
    QVector<DPMRect *> result;
    int i,n=rects.size();
    for(i=0;i<n;i++)
    {
        if(rects[i]!=pixmap)
        {
            DPMRect * rect=(DPMRect *)(rects[i]);
            if(filter.value(rect->category))
            {
                result.push_back(rect);
            }
        }
    }
    return result;
}

void DPMModifierWidget::keyPressEvent(QKeyEvent *event)
{
    if(event->key()==Qt::Key_Return)
    {
        emit signalNext();
    }
    else
    {
        QGraphicsView::keyPressEvent(event);
    }
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
            int tmpcount=0;
            QString tmpcategory;
            for(iter=filter.begin();iter!=filter.end();iter++)
            {
                if(iter.value())
                {
                    menu.addAction(iter.key());
                    tmpcount++;
                    tmpcategory=iter.key();
                }
            }
            if(tmpcount==1)
            {
                QString category=tmpcategory;
                int id=idcount.value(category);
                QColor color=colortable.value(category);
                QPointF pos=mapToScene(event->pos());
                DPMRect * rect=new DPMRect(category,id--,color,pos.x(),pos.y(),100,100,pixmap);
                //scene->addItem(rect);
                idcount.remove(category);
                idcount.insert(category,id);
            }
            else if(tmpcount>1)
            {
                QAction * selitem=menu.exec(QCursor::pos());
                if(selitem)
                {
                    QString category=selitem->text();
                    int id=idcount.value(category);
                    QColor color=colortable.value(category);
                    QPointF pos=mapToScene(event->pos());
                    DPMRect * rect=new DPMRect(category,id--,color,pos.x(),pos.y(),100,100,pixmap);
                    //scene->addItem(rect);
                    idcount.remove(category);
                    idcount.insert(category,id);
                }
            }
            return;
        }
        else if(item!=NULL)
        {
            QGraphicsRectItem * rect=static_cast<QGraphicsRectItem *>(item);
            rect->setPen(QPen(Qt::red, 3));
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
                    slotMoveForward(rect);
                }
                else if(selitem->text()=="Bottom Level")
                {
                    slotMoveBackward(rect);
                }
            }
        }
    }
    QGraphicsView::mousePressEvent(event);
}

DPMController::DPMController(QWidget *parent)
    : QWidget(parent)
{
    layout=new QVBoxLayout;
    this->setLayout(layout);
}

void DPMController::setCategories(QStringList categories)
{
    int i,n=layout->count();
    for(i=n-1;i>=0;i--)
    {
        QCheckBox * checker=(QCheckBox *)(layout->itemAt(i)->widget());
        layout->removeWidget(checker);
        delete checker;
    }
    int colornum=230;
    cv::Mat graymap(1,colornum,CV_8UC1);
    for(i=0;i<colornum;i++)
    {
        graymap.at<uchar>(i)=i;
    }
    cv::Mat colormap;
    cv::applyColorMap(graymap,colormap,cv::COLORMAP_RAINBOW);
    n=categories.size();
    for(i=0;i<n;i++)
    {
        QCheckBox * checker=new QCheckBox(categories.at(i));
        checker->setChecked(1);
        int colorid=int(i*double(colornum-1)/n);
        cv::Vec3b color=colormap.at<cv::Vec3b>(colorid);
        checker->setStyleSheet(QString("QCheckBox { background-color: rgba(%1, %2, %3, 50%) }").arg(color.val[0]).arg(color.val[1]).arg(color.val[2]));
        layout->addWidget(checker);
        connect(checker,SIGNAL(stateChanged(int)),this,SLOT(slotSetFilter(int)));
    }
}

void DPMController::slotSetFilter(int state)
{
    Q_UNUSED(state);
    int i,n=layout->count();
    QMap<QString,bool> result;
    for(i=0;i<n;i++)
    {
        QCheckBox * checker=(QCheckBox *)(layout->itemAt(i)->widget());
        result.insert(checker->text(),checker->isChecked());
    }
    emit signalSetFilter(result);
}
