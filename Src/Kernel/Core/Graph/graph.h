#ifndef GRAPH_H
#define GRAPH_H

#include<Core/Node/node.h>

namespace RobotSDK
{

class Graph : public QObject
{
public:
    Graph(QObject * parent=0);
    ~Graph();
private:
    typedef Node *(*generateNodePtr)(QString libraryFileName, QString configFileName, QString nodeClass, QString nodeName, QString exName);
    generateNodePtr generateNode;
private:
    QMap< QString, QPair< std::shared_ptr< QThread >, Node * > > _nodes;
    QMultiMap< QPair< QString, QString >, QPair< uint, uint > > _edges;
public slots:
    void addNode(QString nodeFullName, QString libraryFileName, QString configFileName);
    void removeNode(QString nodeFullName);
    void clearNodes();
public slots:
    void addEdge(QString outputNodeFullName, uint outputPortID, QString inputNodeFullName, uint inputPortID);
    void removeEdge(QString outputNodeFullName, uint outputPortID, QString inputNodeFullName, uint inputPortID);
    void removeEdgeByOutputPort(QString outputNodeFullName, uint outputPortID);
    void removeEdgeByInputPort(QString inputNodeFullName, uint inputPortID);
    void clearEdges();
public slots:
    void openAllNode();
    void closeAllNode();
};

}



#endif // GRAPH_H
