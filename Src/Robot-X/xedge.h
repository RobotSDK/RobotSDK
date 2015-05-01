#ifndef XEDGE_H
#define XEDGE_H

#include<QGraphicsPathItem>
#include<QGraphicsSceneMouseEvent>
#include<QMenu>
#include<QAction>
#include<QObject>
#include<QPen>

namespace RobotX
{

class XEdge : public QObject, public QGraphicsPathItem
{
    Q_OBJECT
    friend class XGraph;
public:
    XEdge(QGraphicsItem * parent=0);
    ~XEdge();
public:
    QString outputnodefullname;
    uint outputportid;
    QString inputnodefullname;
    uint inputportid;
signals:
    void signalRemoveEdge(QString outputNodeFullName, uint outputPortID, QString inputNodeFullName, uint inputPortID);
protected:
    void mousePressEvent(QGraphicsSceneMouseEvent * event);
    void mouseReleaseEvent(QGraphicsSceneMouseEvent * event);
};

}

#endif // XEDGE_H
