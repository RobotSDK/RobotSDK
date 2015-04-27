#ifndef DPMMODIFIERWIDGETS_H
#define DPMMODIFIERWIDGETS_H

#include"DPMDetector.h"
#include<QGraphicsView>
#include<QGraphicsPixmapItem>
#include<QGraphicsRectItem>
#include<QPushButton>
#include<QImage>
#include<QMenu>
#include<QAction>
#include<QMouseEvent>
#include<QGraphicsSceneMouseEvent>

#include<RobotSDK.h>

class RobotSDK_EXPORT DPMViewer : public QGraphicsView
{
    Q_OBJECT
public:
    DPMViewer(QWidget * parent=0);
protected:
    void mousePressEvent(QMouseEvent *event);
protected:
    QGraphicsScene * scene;
    QGraphicsPixmapItem * pixmap;
public slots:
    void slotDeleteRect(QGraphicsRectItem * rect);
public:
    void clear();
    void addPixmap(QImage & image);
    void addRect(qreal x, qreal y, qreal width, qreal height);
    QVector<QRectF> getRects();
};

class RobotSDK_EXPORT DPMRect : public QGraphicsRectItem
{
public:
    DPMRect(qreal x, qreal y, qreal width, qreal height, QGraphicsItem * parent=0);
protected:
    bool edgeflag;
    Qt::Edge edge;
    bool cornerflag;
    Qt::Corner corner;
    bool resizeflag;
    QRectF orirect;
    qreal oriz;
protected:
    void mousePressEvent(QGraphicsSceneMouseEvent *event);
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event);
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
    void hoverEnterEvent(QGraphicsSceneHoverEvent *event);
    void hoverLeaveEvent(QGraphicsSceneHoverEvent *event);
    void hoverMoveEvent(QGraphicsSceneHoverEvent *event);
};

#endif // DPMMODIFIERWIDGETS_H
