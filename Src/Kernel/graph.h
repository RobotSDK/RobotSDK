#ifndef GRAPH_H
#define GRAPH_H

#include<node.h>

namespace RobotSDK
{

class Graph : public QObject
{
    Q_OBJECT
public:
    Graph(QObject * parent=0);
    ~Graph();
protected:
    void registerTransferData();
private:
    typedef Node *(*generateNodePtr)(QString libraryFileName, QString configFileName, QString nodeFullName);
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
    void changeNodeExName(QString oldNodeFullName, QString newNodeFullName);
    void changeNodeLibrary(QString nodeFullName, QString libraryFileName);
    void changeNodeConfigFile(QString nodeFullName, QString configFileName);
signals:
    void changeNodeResult(bool successFlag, const Node * node);
public slots:
    void openNode(QString nodeFullName);
    void closeNode(QString nodeFullName);
    void openAllNode();
    void closeAllNode();
public slots:
    void showWidget(QString nodeFullName);
    void hideWidget(QString nodeFullName);
    void showAllWidget();
    void hideAllWidget();
public:
    QWidget * const switcherpanel=new QWidget;
protected:
    QVBoxLayout * nodeswitcher;
public:
    const Node * getNode(QString nodeFullName);
    const QWidget * getNodeWidget(QString nodeFullName);
    bool contains(QString nodeFullName);
};

}



#endif // GRAPH_H
