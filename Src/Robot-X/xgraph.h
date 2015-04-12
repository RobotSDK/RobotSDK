#ifndef XGRAPH_H
#define XGRAPH_H

#include<QGraphicsView>

class XGraph : public QGraphicsView
{
public:
    XGraph();
    ~XGraph();
protected slots:
    void slotNodeUpdate(QString oldNodeFullName, QString newNodeFullName);
};

#endif // XGRAPH_H
