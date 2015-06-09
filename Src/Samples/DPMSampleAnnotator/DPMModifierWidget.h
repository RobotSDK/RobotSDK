#ifndef DPMMODIFIERWIDGET_H
#define DPMMODIFIERWIDGET_H

#include"DPMReceiver.h"
#include"ROSBagLoader.h"
#include<QWidget>
#include<QGraphicsPixmapItem>
#include<QGraphicsRectItem>
#include<QGraphicsScene>
#include<QGraphicsView>
#include<QPushButton>
#include<QImage>
#include<QMenu>
#include<QAction>
#include<QMouseEvent>
#include<QGraphicsSceneMouseEvent>
#include<QMap>
#include<QColor>
#include<QPen>

class RobotSDK_EXPORT DPMRect : public QGraphicsRectItem
{
public:
    DPMRect(QString rectCategory, uint rectID, QColor rectColor, qreal x, qreal y, qreal width, qreal height, QGraphicsItem * parent=0);
public:
    QString category;
    uint id;
    QColor color;
protected:
    bool edgeflag;
    Qt::Edge edge;
    bool cornerflag;
    Qt::Corner corner;
    bool resizeflag;
    QRectF orirect;
protected:
    void mousePressEvent(QGraphicsSceneMouseEvent *event);
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event);
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
    void hoverEnterEvent(QGraphicsSceneHoverEvent *event);
    void hoverLeaveEvent(QGraphicsSceneHoverEvent *event);
    void hoverMoveEvent(QGraphicsSceneHoverEvent *event);
};

class RobotSDK_EXPORT DPMModifierWidget : public QGraphicsView
{
    Q_OBJECT
public:
    explicit DPMModifierWidget(QStringList categories, QWidget *parent = 0);
protected:
    void mousePressEvent(QMouseEvent * event);
protected:
    QGraphicsScene * scene;
    QGraphicsPixmapItem * pixmap;
    uint idcount;
public:
    QMap<QString, QColor> colortable;
    QMap<QString, bool> filter;
    QMap<QString, int> idcount;
public slots:
    void slotDeleteRect(QGraphicsRectItem * rect);
    void slotMoveForward(QGraphicsRectItem * rect);
    void slotMoveBackward(QGraphicsRectItem * rect);
public:
    void clear();
    void addPixmap(QImage & image);
    void addRect(QString rectCategory, uint rectID, qreal x, qreal y, qreal width, qreal height);
    QVector<DPMRect *> getRects();
};

#endif // DPMMODIFIERWIDGET_H
