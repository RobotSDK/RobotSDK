#ifndef DPMMODIFIERWIDGET_H
#define DPMMODIFIERWIDGET_H

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
#include<QCheckBox>
#include<opencv2/opencv.hpp>

#include<RobotSDK.h>
namespace RobotSDK_Module
{

class RobotSDK_EXPORT DPMRect : public QGraphicsRectItem
{
public:
    DPMRect(QString rectCategory, int rectID, QColor rectColor, qreal x, qreal y, qreal width, qreal height, QGraphicsItem * parent=0);
public:
    QString category;
    int id;
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
    explicit DPMModifierWidget(QWidget *parent = 0);
    void setCategories(QStringList categories);
protected:
    void mousePressEvent(QMouseEvent * event);
protected:
    QGraphicsScene * scene;
    QGraphicsPixmapItem * pixmap;
public:
    QMap<QString, QColor> colortable;
    QMap<QString, bool> filter;
    QMap<QString, int> idcount;
signals:
    void signalNext();
public slots:
    void slotDeleteRect(QGraphicsRectItem * rect);
    void slotMoveForward(QGraphicsRectItem * rect);
    void slotMoveBackward(QGraphicsRectItem * rect);
    void slotSetFilter(QMap<QString, bool> categoryfilter);
public:
    void clear();
    void addPixmap(QImage & image);
    void addRect(QString rectCategory, int rectID, qreal x, qreal y, qreal width, qreal height);
    QVector<DPMRect *> getRects();
protected:
    void keyPressEvent(QKeyEvent *event);
};

class RobotSDK_EXPORT DPMController : public QWidget
{
    Q_OBJECT
public:
    DPMController(QWidget * parent = 0);
    void setCategories(QStringList categories);
protected:
    QVBoxLayout * layout;
signals:
    void signalSetFilter(QMap<QString, bool> categoryfilter);
public slots:
    void slotSetFilter(int state);
};

}

#endif // DPMMODIFIERWIDGET_H
