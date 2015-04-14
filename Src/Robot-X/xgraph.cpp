#include "xgraph.h"

XGraph::XGraph(QObject *parent)
    : QGraphicsScene(parent)
{
    graph=new RobotSDK::Graph;

    connect(this,SIGNAL(signalAddEdge(QString,uint,QString,uint)),graph,SLOT(addEdge(QString,uint,QString,uint)));
    connect(this,SIGNAL(signalRemoveNode(QString)),graph,SLOT(removeNode(QString)));
    connect(this,SIGNAL(signalRemoveEdge(QString,uint,QString,uint)),graph,SLOT(removeEdge(QString,uint,QString,uint)));
    connect(this,SIGNAL(signalRemoveEdgeByOutputPort(QString,uint)),graph,SLOT(removeEdgeByOutputPort(QString,uint)));
    connect(this,SIGNAL(signalRemoveEdgeByInputPort(QString,uint)),graph,SLOT(removeEdgeByInputPort(QString,uint)));

    _context=gvContext();
    _graph=_agopen("Robot-X");
    _agset(_graph, "overlap", "prism");
    _agset(_graph, "splines", "true");
    _agset(_graph, "nodesep", "2");
    _agset(_graph, "ranksep", "2");
    _agset(_graph, "rankdir", "LR");

    qRegisterMetaType<XPort::PORTTYPE>("XPort::PORTTYPE");
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
    QStringList checkname=nodeFullName.split("::",QString::SkipEmptyParts);
    if(checkname.size()<2||checkname.size()>3)
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
    connect(node,SIGNAL(signalRemovePort(XPort::PORTTYPE,QString,uint)),this,SLOT(slotRemovePort(XPort::PORTTYPE,QString,uint)),Qt::QueuedConnection);
    connect(node,SIGNAL(signalResetPortNum(QString)),this,SLOT(slotResetPortNum(QString)));
    this->addItem(node);
    nodes.insert(nodeFullName,node);

    Agnode_t * tmpnode=_agnode(_graph, nodeFullName,1);
    _nodes.insert(node,tmpnode);
    _agset(tmpnode,"shape","record");
    slotResetPortNum(nodeFullName);
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
            XEdge * edge=candedgelist.at(i).second;
            edge->setToolTip(QString("%1~Port_%2 -> %3~Port_%4").arg(edge->outputnodefullname)
                             .arg(edge->outputportid).arg(edge->inputnodefullname).arg(edge->inputportid));
        }

        agdelete(_graph,_nodes[node]);
        _nodes.remove(node);

        Agnode_t * tmpnode=_agnode(_graph, newNodeFullName,1);        
        _nodes.insert(node,tmpnode);

        _agset(tmpnode,"shape","record");
        _agset(tmpnode,"height",QString("%1").arg(node->geometry().height()/DotDefaultDPI));
        _agset(tmpnode,"width",QString("%1").arg(node->geometry().width()/DotDefaultDPI));

        slotResetPortNum(newNodeFullName);

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
        if(graph->contains(nodeFullName))
        {
            emit signalRemoveNode(nodeFullName);
        }

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
    if(graph->contains(outputNodeFullName)&&graph->contains(inputNodeFullName))
    {
        emit signalAddEdge(outputNodeFullName,outputPortID,inputNodeFullName,inputPortID);
    }
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
        edge->setToolTip(QString("%1~Port_%2 -> %3~Port_%4").arg(edge->outputnodefullname)
                         .arg(edge->outputportid).arg(edge->inputnodefullname).arg(edge->inputportid));
        connect(edge,SIGNAL(signalRemoveEdge(QString,uint,QString,uint)),this,SLOT(slotRemoveEdge(QString,uint,QString,uint)),Qt::QueuedConnection);
        this->addItem(edge);
        edges.insert(QPair< QString, QString >(outputNodeFullName, inputNodeFullName), edge);

        XNode * source=nodes[edge->outputnodefullname];
        XNode * target=nodes[edge->inputnodefullname];
        QString outputport=QString("out_%1:e").arg(edge->outputportid);
        QString inputport=QString("in_%1:w").arg(edge->inputportid);
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
        if(graph->contains(outputNodeFullName)&&graph->contains(inputNodeFullName))
        {
            emit signalRemoveEdge(outputNodeFullName,outputPortID,inputNodeFullName,inputPortID);
        }

        edges.remove(QPair< QString, QString >(outputNodeFullName, inputNodeFullName), candedges.at(i));
        this->removeItem(candedges.at(i));
        delete candedges[i];
        agdelete(_graph,_edges[candedges[i]]);
        _edges.remove(candedges[i]);

        slotApplyLayout();
    }
}

void XGraph::slotRemovePort(XPort::PORTTYPE portType, QString nodeFullName, uint portID)
{
    if(nodes.contains(nodeFullName))
    {
        if(graph->contains(nodeFullName))
        {
            switch (portType) {
            case XPort::PORTTYPE::InputPort:
                emit signalRemoveEdgeByInputPort(nodeFullName,portID);
                break;
            case XPort::PORTTYPE::OutputPort:
                emit signalRemoveEdgeByOutputPort(nodeFullName,portID);
                break;
            default:
                break;
            }
        }

        QMultiMap< QPair< QString, QString >, XEdge * >::const_iterator edgeiter;
        QList< QPair< QPair< QString, QString >, XEdge * > > candedgelist;
        for(edgeiter=edges.begin();edgeiter!=edges.end();edgeiter++)
        {
            if((portType==XPort::PORTTYPE::OutputPort&&edgeiter.key().first==nodeFullName)
                    ||(portType==XPort::PORTTYPE::InputPort&&edgeiter.key().second==nodeFullName))
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
            agdelete(_graph,_edges[candedgelist[i].second]);
            _edges.remove(candedgelist[i].second);
        }
        slotApplyLayout();
    }
}

void XGraph::slotResetPortNum(QString nodeFullName)
{
    Agnode_t * tmpnode=_nodes[nodes[nodeFullName]];
    XNode * node=nodes[nodeFullName];
    uint i;
    QString inputlabel=QString("Input");
    for(i=0;i<node->_inputportnum;i++)
    {
        inputlabel+=QString(" | <in_%1> Port_%1").arg(i);
    }
    QString outputlabel=QString("Output");
    for(i=0;i<node->_outputportnum;i++)
    {
        outputlabel+=QString(" | <out_%1> Port_%1").arg(i);
    }
    QString nodelabel=QString("{{%1} | %2 | {%3}}").arg(inputlabel).arg(nodeFullName).arg(outputlabel);
    _agset(tmpnode,"label",nodelabel);
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

void XGraph::slotHandleMenu()
{
    QMenu menu;
    menu.addAction("Add a Virtual Node...");
    menu.addAction("Add a Real Node...");
    menu.addSeparator();
    menu.addAction("Load Graph...");
    menu.addAction("Save Graph...");
    menu.addSeparator();
    menu.addAction("Open All Nodes");
    menu.addAction("Close All Nodes");
    menu.addAction("Show All Widgets");
    menu.addAction("Hide All Widgets");
    menu.addSeparator();
    menu.addAction("Export Graph Image...");
    menu.addAction("Export Dot File...");
    menu.addSeparator();
    menu.addAction("Clean Graph");
    QAction * selecteditem=menu.exec(QCursor::pos());
    if(selecteditem)
    {
        if(selecteditem->text()==QString("Add a Virtual Node..."))
        {
            QString nodefullname=QInputDialog::getText(NULL,"Add a Virtual Node","Node Full Name (NodeClass::NodeName[::ExName])");
            if(nodefullname.size()>0)
            {
                slotAddNode(nodefullname);
            }
        }
        else if(selecteditem->text()==QString("Add a Real Node..."))
        {
            QString nodefullname=QInputDialog::getText(NULL,"Add a Real Node","Node Full Name (NodeClass::NodeName[::ExName])");
            if(nodefullname.size()>0)
            {
#ifdef Q_OS_LINUX
                QString libraryfilename=QFileDialog::getOpenFileName(NULL,"Add a Real Node",QString(),QString("Shared Library (*.so)"));
#endif
#ifdef Q_OS_WIN32
                QString libraryfilename=QFileDialog::getOpenFileName(NULL,"Add a Real Node",QString(),QString("Shared Library (*.dll)"));
#endif
                if(libraryfilename.size()>0)
                {
                    slotAddNode(nodefullname,libraryfilename);
                }
            }
        }
        else if(selecteditem->text()==QString("Load Graph..."))
        {
            QString filename=QFileDialog::getOpenFileName(NULL,"Load Graph",QString(),QString("X (*.x)"));
            if(filename.size()>0)
            {
                slotLoadGraph(filename);
            }
        }
        else if(selecteditem->text()==QString("Save Graph..."))
        {
            QString filename=QFileDialog::getSaveFileName(NULL,"Save Graph",QString(),QString("X (*.x)"));
            if(filename.size()>0)
            {
                slotSaveGraph(filename);
            }
        }
        else if(selecteditem->text()==QString("Open All Nodes"))
        {
            graph->openAllNode();
        }
        else if(selecteditem->text()==QString("Close All Nodes"))
        {
            graph->closeAllNode();
        }
        else if(selecteditem->text()==QString("Show All Widgets"))
        {
            graph->showAllWidget();
            QMap< QString, XNode * >::const_iterator nodeiter;
            for(nodeiter=nodes.begin();nodeiter!=nodes.end();nodeiter++)
            {
                if(graph->contains(nodeiter.key()))
                {
                    nodeiter.value()->showwidget->setStyleSheet("QPushButton {background-color: green; color: black;}");
                    nodeiter.value()->showwidget->setText("Hide Widget");
                }
            }
        }
        else if(selecteditem->text()==QString("Hide All Widgets"))
        {
            graph->hideAllWidget();
            QMap< QString, XNode * >::const_iterator nodeiter;
            for(nodeiter=nodes.begin();nodeiter!=nodes.end();nodeiter++)
            {
                if(graph->contains(nodeiter.key()))
                {
                    nodeiter.value()->showwidget->setStyleSheet("QPushButton {background-color: red; color: black;}");
                    nodeiter.value()->showwidget->setText("Show Widget");
                }
            }
        }
        else if(selecteditem->text()==QString("Export Graph Image..."))
        {
            QString filename=QFileDialog::getSaveFileName(NULL,"Export Graph Image",QString(),QString("Image (*.png)"));
            if(filename.size()>0)
            {
                _gvLayout(_context,_graph,"dot");
                gvRenderFilename(_context,_graph,"png",filename.toUtf8().data());
                gvFreeLayout(_context,_graph);
            }
        }
        else if(selecteditem->text()==QString("Export Dot File..."))
        {
            QString filename=QFileDialog::getSaveFileName(NULL,"Export Dot File",QString(),QString("Dot (*.dot)"));
            if(filename.size()>0)
            {
                _gvLayout(_context,_graph,"dot");
                gvRenderFilename(_context,_graph,"dot",filename.toUtf8().data());
                gvFreeLayout(_context,_graph);
            }
        }
        else if(selecteditem->text()==QString("Clean Graph"))
        {
            QStringList nodelist=nodes.keys();
            uint i,n=nodelist.size();
            for(i=0;i<n;i++)
            {
                slotRemoveNode(nodelist.at(i));
            }
        }
    }
}

void XGraph::slotLoadGraph(QString xFileName)
{
    QFile file(xFileName);
    if(file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        while(!file.atEnd())
        {
            QString line=file.readLine();
            if(line.size()>0)
            {
                QStringList data=line.split(QString(","));
                if(data.at(0).trimmed()=="N"&&data.size()==6)
                {
                    slotAddNode(data.at(1).trimmed(),data.at(2).trimmed(),data.at(3).trimmed());
                    nodes[data.at(1).trimmed()]->slotResetPortNum("Input",data.at(4).toUInt());
                    nodes[data.at(1).trimmed()]->slotResetPortNum("Output",data.at(5).toUInt());
                }
                else if(data.at(0).trimmed()=="E"&&data.size()==5)
                {
                    slotAddEdge(data.at(1),data.at(2).toUInt(),data.at(3),data.at(4).toUInt());
                }
            }
        }
        file.close();
    }
}

void XGraph::slotSaveGraph(QString xFileName)
{
    QFile file(xFileName);
    if(file.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        QTextStream stream(&file);
        uint i,n;
        QList< XNode * > nodelist=nodes.values();
        n=nodelist.size();
        for(i=0;i<n;i++)
        {
            if(graph->contains(nodelist.at(i)->nodefullname->text()))
            {
                stream<<QString("N,%1,%2,%3,%4,%5\n")
                        .arg(nodelist.at(i)->nodefullname->text())
                        .arg(nodelist.at(i)->libraryfilename->text())
                        .arg(nodelist.at(i)->configfilename->text())
                        .arg(nodelist.at(i)->_inputportnum)
                        .arg(nodelist.at(i)->_outputportnum);
            }
            else
            {
                stream<<QString("N,%1,%2,%3,%4,%5\n")
                        .arg(nodelist.at(i)->nodefullname->text())
                        .arg(" ")
                        .arg(nodelist.at(i)->configfilename->text())
                        .arg(nodelist.at(i)->_inputportnum)
                        .arg(nodelist.at(i)->_outputportnum);
            }
        }
        QList< XEdge *> edgelist=edges.values();
        n=edgelist.size();
        for(i=0;i<n;i++)
        {
            stream<<QString("E,%1,%2,%3,%4\n")
                    .arg(edgelist.at(i)->outputnodefullname)
                    .arg(edgelist.at(i)->outputportid)
                    .arg(edgelist.at(i)->inputnodefullname)
                    .arg(edgelist.at(i)->inputportid);
        }
        file.close();
    }
}
