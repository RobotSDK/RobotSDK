#include "xgraph.h"

XGraph::XGraph(QObject *parent)
    : QGraphicsScene(parent)
{
    graph=new RobotSDK::Graph;
    connect(this,SIGNAL(signalRemoveEdge(QString,uint,QString,uint)),graph,SLOT(removeEdge(QString,uint,QString,uint)));

    _context=gvContext();
    _graph=_agopen("Robot-X");
    _agset(_graph, "splines", "true");
    _agset(_graph, "nodesep", "0.4");
}

XGraph::~XGraph()
{
    gvFreeLayout(_context,_graph);
    agclose(_graph);
    gvFreeContext(_context);
    delete graph;
}

void XGraph::slotAddNode(QString nodeFullName, QString libraryFileName, QString configFileName)
{
    if(nodes.contains(nodeFullName))
    {
        return;
    }
    if(libraryFileName.size()>0)
    {
        graph->addNode(nodeFullName,libraryFileName,configFileName);
    }
    XNode * node=new XNode(graph,nodeFullName);
    connect(node,SIGNAL(signalResize(QString,QSizeF)),this,SLOT(slotResize(QString,QSizeF)));
    connect(node,SIGNAL(signalUpdateNode(QString,QString)),this,SLOT(slotUpdateNode(QString,QString)),Qt::QueuedConnection);
    connect(node,SIGNAL(signalRemoveNode(QString)),this,SLOT(slotRemoveNode(QString)),Qt::QueuedConnection);
    connect(node,SIGNAL(signalAddEdge(QString,uint,QString,uint)),this,SLOT(slotAddEdge(QString,uint,QString,uint)),Qt::QueuedConnection);
    this->addItem(node);
    nodes.insert(nodeFullName,node);

    Agnode_t * tmpnode=_agnode(_graph, nodeFullName,1);
    _agset(tmpnode,"shape","record");
    _agset(tmpnode,"height",QString("%1").arg(node->geometry().height()/DotDefaultDPI));
    _agset(tmpnode,"width",QString("%1").arg(node->geometry().width()/DotDefaultDPI));

    uint i;
    QString inputlabel=QString("Input Ports");
    for(i=0;i<node->_inputportnum;i++)
    {
        inputlabel+=QString(" | <in_%1> #%1").arg(i);
    }
    QString outputlabel=QString("Output Ports");
    for(i=0;i<node->_outputportnum;i++)
    {
        outputlabel+=QString(" | <out_%1> #%1").arg(i);
    }
    QString nodelabel=QString("{%1} | %2 | {%3}").arg(inputlabel).arg(nodeFullName).arg(outputlabel);
    _agset(tmpnode,"label",nodelabel);

    _nodes.insert(node,tmpnode);
}

void XGraph::slotResize(QString nodeFullName, QSizeF newSize)
{
    Agnode_t * node=_nodes[nodes[nodeFullName]];
    _agset(node,"height",QString("%1").arg(newSize.height()/DotDefaultDPI));
    _agset(node,"width",QString("%1").arg(newSize.width()/DotDefaultDPI));

    slotApplyLayout();
}

void XGraph::slotUpdateNode(QString oldNodeFullName, QString newNodeFullName)
{
    if(nodes.contains(oldNodeFullName))
    {
        XNode * node=nodes[oldNodeFullName];
        nodes.remove(oldNodeFullName);
        nodes.insert(newNodeFullName,node);
        QMultiMap< QPair< QString, QString >, XEdge * >::const_iterator edgeiter;
        QList< QPair< QPair< QString, QString >, XEdge * > > candedgelist;
        for(edgeiter=edges.begin();edgeiter!=edges.end();edgeiter++)
        {
            if(edgeiter.key().first==oldNodeFullName||edgeiter.key().second==oldNodeFullName)
            {
                candedgelist.push_back(QPair< QPair< QString, QString >, XEdge * >(edgeiter.key(),edgeiter.value()));
            }
        }
        uint i,n=candedgelist.size();
        for(i=0;i<n;i++)
        {
            edges.remove(candedgelist.at(i).first,candedgelist.at(i).second);
            if(candedgelist.at(i).first.first==oldNodeFullName)
            {
                candedgelist[i].first.first=newNodeFullName;
                candedgelist[i].second->outputnodefullname=newNodeFullName;
            }
            if(candedgelist.at(i).first.second==oldNodeFullName)
            {
                candedgelist[i].first.second=newNodeFullName;
                candedgelist[i].second->inputnodefullname=newNodeFullName;
            }
            edges.insert(candedgelist.at(i).first,candedgelist.at(i).second);
        }

        agdelete(_graph,_nodes[node]);
        _nodes.remove(node);

        Agnode_t * tmpnode=_agnode(_graph, newNodeFullName,1);
        _agset(tmpnode,"shape","record");
        _agset(tmpnode,"height",QString("%1").arg(node->geometry().height()/DotDefaultDPI));
        _agset(tmpnode,"width",QString("%1").arg(node->geometry().width()/DotDefaultDPI));

        QString inputlabel=QString("Input Ports");
        for(i=0;i<node->_inputportnum;i++)
        {
            inputlabel+=QString(" | <in_%1> #%1").arg(i);
        }
        QString outputlabel=QString("Output Ports");
        for(i=0;i<node->_outputportnum;i++)
        {
            outputlabel+=QString(" | <out_%1> #%1").arg(i);
        }
        QString nodelabel=QString("{%1} | %2 | {%3}").arg(inputlabel).arg(newNodeFullName).arg(outputlabel);
        _agset(tmpnode,"label",nodelabel);

        _nodes.insert(node,tmpnode);

        for(i=0;i<n;i++)
        {
            _edges.remove(candedgelist.at(i).second);
            XNode * source=nodes[candedgelist.at(i).first.first];
            XNode * target=nodes[candedgelist.at(i).first.second];
            QString outputport=QString("out_%1:e").arg(candedgelist.at(i).second->outputportid);
            QString inputport=QString("in_%1:w").arg(candedgelist.at(i).second->inputportid);
            Agedge_t * tmpedge=_agedge(_graph,_nodes[source],_nodes[target],QString(),1);
            _agset(tmpedge,"headport",outputport);
            _agset(tmpedge,"tailport",inputport);
            _edges.insert(candedgelist.at(i).second,tmpedge);
        }

        slotApplyLayout();
    }
}

void XGraph::slotRemoveNode(QString nodeFullName)
{
    if(nodes.contains(nodeFullName))
    {
        XNode * node=nodes[nodeFullName];
        this->removeItem(node);
        nodes.remove(nodeFullName);
        delete node;
        QMultiMap< QPair< QString, QString >, XEdge * >::const_iterator edgeiter;
        QList< QPair< QPair< QString, QString >, XEdge * > > candedgelist;
        for(edgeiter=edges.begin();edgeiter!=edges.end();edgeiter++)
        {
            if(edgeiter.key().first==nodeFullName||edgeiter.key().second==nodeFullName)
            {
                candedgelist.push_back(QPair< QPair< QString, QString >, XEdge * >(edgeiter.key(),edgeiter.value()));
            }
        }
        uint i,n=candedgelist.size();
        for(i=0;i<n;i++)
        {
            edges.remove(candedgelist.at(i).first,candedgelist.at(i).second);
            this->removeItem(candedgelist.at(i).second);
            delete candedgelist[i].second;
        }

        agdelete(_graph,_nodes[node]);
        _nodes.remove(node);
        for(i=0;i<n;i++)
        {
            _edges.remove(candedgelist.at(i).second);
        }

        slotApplyLayout();
    }
}

void XGraph::slotAddEdge(QString outputNodeFullName, uint outputPortID, QString inputNodeFullName, uint inputPortID)
{
    QList<XEdge *> candedges=edges.values(QPair< QString, QString >(outputNodeFullName, inputNodeFullName));
    uint i,n=candedges.size();
    for(i=0;i<n;i++)
    {
        if(candedges.at(i)->outputportid==outputPortID&&candedges.at(i)->inputportid==inputPortID)
        {
            break;
        }
    }
    if(i==n)
    {
        XEdge * edge=new XEdge;
        edge->outputnodefullname=outputNodeFullName;
        edge->outputportid=outputPortID;
        edge->inputnodefullname=inputNodeFullName;
        edge->inputportid=inputPortID;
        connect(edge,SIGNAL(signalRemoveEdge(QString,uint,QString,uint)),this,SLOT(slotRemoveEdge(QString,uint,QString,uint)),Qt::QueuedConnection);
        this->addItem(edge);
        edges.insert(QPair< QString, QString >(outputNodeFullName, inputNodeFullName), edge);

        XNode * source=nodes[edge->outputnodefullname];
        XNode * target=nodes[edge->inputnodefullname];
        QString outputport=QString("out_%1:e").arg(edge->outputportid);
        QString inputport=QString("in_%1:w").arg(edge->inputportid);
        QString edgename=QString("%1~~%2~~%3~~%4")
                .arg(edge->outputnodefullname)
                .arg(outputport)
                .arg(edge->inputnodefullname)
                .arg(inputport);
        Agedge_t * tmpedge=_agedge(_graph,_nodes[source],_nodes[target],QString(),1);
        _agset(tmpedge,"tailport",outputport);
        _agset(tmpedge,"headport",inputport);
        _edges.insert(edge,tmpedge);
        slotApplyLayout();
    }
}

void XGraph::slotRemoveEdge(QString outputNodeFullName, uint outputPortID, QString inputNodeFullName, uint inputPortID)
{
    QList<XEdge *> candedges=edges.values(QPair< QString, QString >(outputNodeFullName, inputNodeFullName));
    uint i,n=candedges.size();
    for(i=0;i<n;i++)
    {
        if(candedges.at(i)->outputportid==outputPortID&&candedges.at(i)->inputportid==inputPortID)
        {
            break;
        }
    }
    if(i<n)
    {
        emit signalRemoveEdge(outputNodeFullName,outputPortID,inputNodeFullName,inputPortID);

        edges.remove(QPair< QString, QString >(outputNodeFullName, inputNodeFullName), candedges.at(i));
        this->removeItem(candedges.at(i));
        delete candedges[i];
        agdelete(_graph,_edges[candedges[i]]);
        _edges.remove(candedges[i]);

        slotApplyLayout();
    }
}

void XGraph::slotApplyLayout()
{
    _gvLayout(_context,_graph,"dot");
    this->setSceneRect(boudingRect());

    uint i,n;

    QList< XNode *> nodelist=nodes.values();
    n=nodelist.size();
    for(i=0;i<n;i++)
    {
        setNodePos(nodelist[i]);
    }
    QList< XEdge *> edgelist=edges.values();
    n=edgelist.size();
    for(i=0;i<n;i++)
    {
        drawEdgePath(edgelist[i]);
    }

    gvRenderFilename(_context,_graph,"png","./out.png");
    gvRenderFilename(_context,_graph,"dot","./out.dot");
    gvFreeLayout(_context,_graph);
}

QRectF XGraph::boudingRect()
{
    boxf rect=GD_bb(_graph);
    return QRectF(QPointF(rect.LL.x,rect.LL.y),QPointF(rect.UR.x,rect.UR.y));
}

 void XGraph::setNodePos(XNode *node)
 {
     boxf rect=GD_bb(_graph);
     pointf pos=ND_coord(_nodes[node]);
     qreal width=ND_width(_nodes[node])*DotDefaultDPI;
     qreal height=ND_height(_nodes[node])*DotDefaultDPI;
     qreal x=pos.x-width/2;
     qreal y=rect.UR.y-pos.y-height/2;
     node->setPos(x,y);
 }

 void XGraph::drawEdgePath(XEdge *edge)
 {
     Agedge_t * tmpedge=_edges[edge];
     const splines * spl=ED_spl(tmpedge);
     QPainterPath path;
     qreal gheight=GD_bb(_graph).UR.y;
     if((spl->list != 0) && (spl->list->size%3 == 1))
     {
         bezier bez = spl->list[0];
         //If there is a starting point, draw a line from it to the first curve point
         if(bez.sflag)
         {
             path.moveTo(bez.sp.x,gheight-bez.sp.y);
             path.lineTo(bez.list[0].x, gheight-bez.list[0].y);
         }
         else
             path.moveTo(bez.list[0].x, gheight-bez.list[0].y);

         //Loop over the curve points
         for(int i=1; i<bez.size; i+=3)
             path.cubicTo(QPointF(bez.list[i].x, gheight-bez.list[i].y), QPointF(bez.list[i+1].x, gheight-bez.list[i+1].y), QPointF(bez.list[i+2].x, gheight-bez.list[i+2].y));

         //If there is an ending point, draw a line to it
         if(bez.eflag)
             path.lineTo(bez.ep.x, gheight-bez.ep.y);
     }
     edge->setPath(path);
 }

 Agraph_t * XGraph::_agopen(QString name)
 {
     return agopen(const_cast<char *>(qPrintable(name)), Agdirected, 0);
 }

QString XGraph::_agget(void *object, QString attr, QString alt)
{
    QString str=agget(object, const_cast<char *>(qPrintable(attr)));
    if(str==QString())
    {
        return alt;
    }
    else
    {
        return str;
    }
}

int XGraph::_agset(void *object, QString attr, QString value)
{
    return agsafeset(object,const_cast<char *>(qPrintable(attr)) ,
                     const_cast<char *>(qPrintable(value)),
                     const_cast<char *>(qPrintable(value)));
}

Agnode_t * XGraph::_agnode(Agraph_t *object, QString name, int flag)
{
    return agnode(object, const_cast<char *>(qPrintable(name)), flag);
}

Agedge_t * XGraph::_agedge(Agraph_t *object, Agnode_t *source, Agnode_t *target, QString name, int flag)
{
    if(name.size()>0)
    {
        return agedge(object,source, target, const_cast<char *>(qPrintable(name)),flag);
    }
    else
    {
        return agedge(object,source, target, NULL,flag);
    }
}

int XGraph::_gvLayout(GVC_t *context, Agraph_t *object, QString layout)
{
    return gvLayout(context,object,const_cast<char *>(qPrintable(layout)));
}
