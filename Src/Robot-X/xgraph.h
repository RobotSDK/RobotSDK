#ifndef XGRAPH_H
#define XGRAPH_H

#include<QGraphicsScene>
#include<QMap>
#include<QMultiMap>
#include<QPair>
#include<QFile>
#include<QTextStream>

#include<gvc.h>

#include"xnode.h"
#include"xedge.h"

namespace RobotX
{

#define DotDefaultDPI 72.0

class XGraph : public QGraphicsScene
{
    Q_OBJECT
public:
    XGraph(QObject * parent=0);
    ~XGraph();
protected:
    Graph * graph;
    QMap< QString, XNode * > nodes;
    QMultiMap< QPair< QString, QString >, XEdge * > edges;
protected:
    GVC_t * _context;
    Agraph_t * _graph;
    QMap< XNode *, Agnode_t * > _nodes;
    QMap< XEdge *, Agedge_t * > _edges;
public slots:
    void slotAddNode(QString nodeFullName, QString libraryFileName=QString(), QString configFileName=QString("Config.xml"));
protected slots:
    void slotResize(QString nodeFullName, QSizeF newSize);
    void slotUpdateNode(QString oldNodeFullName, QString newNodeFullName);
    void slotRemoveNode(QString nodeFullName);
    void slotAddEdge(QString outputNodeFullName, uint outputPortID, QString inputNodeFullName, uint inputPortID);
    void slotRemoveEdge(QString outputNodeFullName, uint outputPortID, QString inputNodeFullName, uint inputPortID);
    void slotRemovePort(XPort::PORTTYPE portType, QString nodeFullName,uint portID);
    void slotResetPortNum(QString nodeFullName);
signals:
    void signalAddEdge(QString outputNodeFullName, uint outputPortID, QString inputNodeFullName, uint inputPortID);
    void signalRemoveNode(QString nodeFullName);
    void signalRemoveEdge(QString outputNodeFullName, uint outputPortID, QString inputNodeFullName, uint inputPortID);
    void signalRemoveEdgeByOutputPort(QString outputNodeFullName, uint outputPortID);
    void signalRemoveEdgeByInputPort(QString inputNodeFullName, uint inputPortID);
public slots:
    void slotApplyLayout();
protected:
    QRectF boudingRect();
    void setNodePos(XNode * node);
    void drawEdgePath(XEdge * edge);
private:
    Agraph_t * _agopen(QString name);
    QString _agget(void * object, QString attr, QString alt=QString());
    int _agset(void * object, QString attr, QString value);
    Agnode_t * _agnode(Agraph_t * object, QString name, int flag);
    Agedge_t * _agedge(Agraph_t * object, Agnode_t * source, Agnode_t * target, QString name, int flag);
    int _gvLayout(GVC_t * context, Agraph_t * object, QString layout);
public slots:
    void slotHandleMenu();
public slots:
    void slotLoadGraph(QString xFileName);
    void slotSaveGraph(QString xFileName);
};

}

#endif // XGRAPH_H
