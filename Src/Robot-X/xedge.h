#ifndef XEDGE_H
#define XEDGE_H

#include<QGraphicsPathItem>

class XEdge : public QGraphicsPathItem
{
public:
    XEdge();
    ~XEdge();
protected:
    QString outputnodefullname;
    uint outputportid;
    QString inputnodefullname;
    uint inputportid;
};

#endif // XEDGE_H
