#include "graph.h"

using namespace RobotSDK;

Graph::Graph(QObject * parent)
    : QObject(parent)
{    
    registerTransferData();

    nodeswitcher=new QVBoxLayout;
    QVBoxLayout * layout=new QVBoxLayout;
    layout->addLayout(nodeswitcher);
    layout->addStretch();
    switcherpanel->setLayout(layout);
}

Graph::~Graph()
{
    closeAllNode();
    clearNodes();
    delete switcherpanel;
}

void Graph::registerTransferData()
{
   REGISTER_TRANSFER_VALUE_TYPE(TRANSFER_NODE_PARAMS_TYPE);
   REGISTER_TRANSFER_VALUE_TYPE(TRANSFER_NODE_VARS_TYPE);
   REGISTER_TRANSFER_VALUE_TYPE(TRANSFER_NODE_DATA_TYPE);
   REGISTER_TRANSFER_VALUE_TYPE(TRANSFER_PORT_PARAMS_TYPE);
   REGISTER_TRANSFER_VALUE_TYPE(TRANSFER_PORT_DATA_TYPE);
   REGISTER_TRANSFER_VALUE_TYPE(PORT_PARAMS_CAPSULE);
   REGISTER_TRANSFER_VALUE_TYPE(PORT_DATA_CAPSULE);
}

void Graph::addNode(QString nodeFullName, QString libraryFileName, QString configFileName)
{
    if(_nodes.contains(nodeFullName))
    {
        qDebug()<<QString("%1 has already existed in the graph.").arg(nodeFullName);
        emit addNodeResult(0,nodeFullName,NULL);
        return;
    }
    QStringList nodenamelist=nodeFullName.split(QString("::"),QString::SkipEmptyParts);
    if(nodenamelist.size()<2||nodenamelist.size()>3)
    {
        qDebug()<<QString("%1 is not a valid node full name (NodeClass::NodeName[::ExName]).");
        emit addNodeResult(0,nodeFullName,NULL);
        return;
    }
    QString nodeClass=nodenamelist.at(0);
    QString nodeName=nodenamelist.at(1);
    QString exName=QString();
    if(nodenamelist.size()==3)
    {
        exName=nodenamelist.at(2);
    }
    QString functionname=QString("%1__%2").arg(nodeClass).arg("generateNode");
    QLibrary library(libraryFileName);
    generateNode=(generateNodePtr)(library.resolve(functionname.toUtf8().data()));
    if(generateNode==NULL)
    {
        qDebug()<<QString("Can not resolve %1 from %2. May lack of USE_DEFAULT_NODE or USE_EXTENDED_NODE in module source code.").arg(functionname).arg(libraryFileName);
        qDebug()<<library.errorString();
        emit addNodeResult(0,nodeFullName,NULL);
        return;
    }
    Node * node=static_cast<Node *>(generateNode(libraryFileName, configFileName, nodeFullName));
    if(node==NULL)
    {
        qDebug()<<QString("Can not build node. May the node's type is not extended from RobotSDK::Node");
        emit addNodeResult(0,nodeFullName,NULL);
        return;
    }
    if(!node->_loadflag||!node->_initializeflag)
    {
        delete node;
        emit addNodeResult(0,nodeFullName,NULL);
        return;
    }
    if(!node->NODE_VARS_ARG->_guithreadflag)
    {
        std::shared_ptr<QThread> thread=std::shared_ptr<QThread>(new QThread);
        connect(thread.get(),SIGNAL(finished()),node,SLOT(deleteLater()));
        _nodes.insert(nodeFullName,QPair< std::shared_ptr< QThread >, Node * >(thread,node));
        node->moveToThread(thread.get());
        thread->start();
    }
    else
    {
        _nodes.insert(nodeFullName,QPair< std::shared_ptr< QThread >, Node * >(std::shared_ptr< QThread >(),node));
    }

    nodeswitcher->addWidget(node->NODE_VARS_ARG->getNodeSwitcher());
    emit addNodeResult(1,nodeFullName,_nodes[nodeFullName].second);
    return;
}

void Graph::removeNode(QString nodeFullName)
{
    if(!_nodes.contains(nodeFullName))
    {
        qDebug()<<QString("%1 does not exist in the graph.").arg(nodeFullName);
        return;
    }

    closeNode(nodeFullName);

    QMap< QString, QPair< std::shared_ptr< QThread >, Node * > >::const_iterator nodeiter;
    for(nodeiter=_nodes.begin();nodeiter!=_nodes.end();nodeiter++)
    {
        _edges.remove(QPair< QString, QString >(nodeFullName,nodeiter.key()));
        _edges.remove(QPair< QString, QString >(nodeiter.key(),nodeFullName));
    }

    nodeswitcher->removeWidget(_nodes[nodeFullName].second->NODE_VARS_ARG->getNodeSwitcher());
    if(_nodes[nodeFullName].first!=NULL)
    {
        _nodes[nodeFullName].first->quit();
        _nodes[nodeFullName].first->wait();
    }
    else
    {
        delete _nodes[nodeFullName].second;
    }
    _nodes.remove(nodeFullName);
    return;
}

void Graph::clearNodes()
{
    QMap< QString, QPair< std::shared_ptr<QThread>, Node * > >::const_iterator nodeiter;
    for(nodeiter=_nodes.begin();nodeiter!=_nodes.end();nodeiter++)
    {
        nodeswitcher->removeWidget(nodeiter.value().second->NODE_VARS_ARG->getNodeSwitcher());
        nodeiter.value().first->exit();
    }
    for(nodeiter=_nodes.begin();nodeiter!=_nodes.end();nodeiter++)
    {
        nodeiter.value().first->wait();
    }
    _nodes.clear();
    _edges.clear();
}

void Graph::addEdge(QString outputNodeFullName, uint outputPortID, QString inputNodeFullName, uint inputPortID)
{
    if(!_nodes.contains(outputNodeFullName)||!_nodes.contains(inputNodeFullName))
    {
        qDebug()<<QString("%1 or %2 do not exist in the graph.").arg(outputNodeFullName).arg(outputNodeFullName);
        return;
    }
    QList< QPair< uint, uint > > edges=_edges.values(QPair< QString, QString >(outputNodeFullName, inputNodeFullName));
    uint i,n=edges.size();
    bool insertflag=1;
    for(i=0;i<n;i++)
    {
        if(edges.at(i).first==outputPortID&&edges.at(i).second==inputPortID)
        {
            insertflag=0;
            break;
        }
    }
    if(insertflag)
    {
        const OutputPort * outputport=_nodes[outputNodeFullName].second->getOutputPort(outputPortID);
        const InputPort * inputport=_nodes[inputNodeFullName].second->getInputPort(inputPortID);
        if(outputport==NULL||inputport==NULL)
        {
            qDebug()<<QString("Port ID is out of range");
            return;
        }
        connect(outputport,OUTPUTPORT_SIGNAL
                ,inputport,INPUTPORT_SLOT,Qt::QueuedConnection);
        _edges.insert(QPair< QString, QString >(outputNodeFullName, inputNodeFullName),QPair< uint, uint >(outputPortID, inputPortID));
    }
}

void Graph::removeEdge(QString outputNodeFullName, uint outputPortID, QString inputNodeFullName, uint inputPortID)
{
    int removednum=_edges.remove(QPair< QString, QString >(outputNodeFullName, inputNodeFullName),QPair< uint, uint >(outputPortID, inputPortID));
    if(removednum>0)
    {
        const OutputPort * outputport=_nodes[outputNodeFullName].second->getOutputPort(outputPortID);
        const InputPort * inputport=_nodes[inputNodeFullName].second->getInputPort(inputPortID);
        disconnect(outputport,OUTPUTPORT_SIGNAL
                ,inputport,INPUTPORT_SLOT);
    }
}

void Graph::removeEdgeByOutputPort(QString outputNodeFullName, uint outputPortID)
{
    if(!_nodes.contains(outputNodeFullName))
    {
        qDebug()<<QString("%1 does not exist in the graph.").arg(outputNodeFullName);
        return;
    }
    const OutputPort * outputport=_nodes[outputNodeFullName].second->getOutputPort(outputPortID);
    if(outputport==NULL)
    {
        qDebug()<<QString("Port ID is out of range");
        return;
    }
    QMultiMap< QPair< QString, QString >, QPair< uint, uint > >::const_iterator edgeiter;
    for(edgeiter=_edges.begin();edgeiter!=_edges.end();edgeiter++)
    {
        if(edgeiter.key().first==outputNodeFullName)
        {
            QString inputNodeFullName=edgeiter.key().second;
            uint inputPortID=edgeiter.value().second;
            const InputPort * inputport=_nodes[inputNodeFullName].second->getInputPort(inputPortID);
            disconnect(outputport,OUTPUTPORT_SIGNAL
                    ,inputport,INPUTPORT_SLOT);
            _edges.remove(edgeiter.key(),edgeiter.value());
            edgeiter--;
        }
    }
}

void Graph::removeEdgeByInputPort(QString inputNodeFullName, uint inputPortID)
{
    if(!_nodes.contains(inputNodeFullName))
    {
        qDebug()<<QString("%1 does not exist in the graph.").arg(inputNodeFullName);
        return;
    }
    const InputPort * inputport=_nodes[inputNodeFullName].second->getInputPort(inputPortID);
    if(inputport==NULL)
    {
        qDebug()<<QString("Port ID is out of range");
        return;
    }
    QMultiMap< QPair< QString, QString >, QPair< uint, uint > >::const_iterator edgeiter;
    for(edgeiter=_edges.begin();edgeiter!=_edges.end();edgeiter++)
    {
        if(edgeiter.key().second==inputNodeFullName)
        {
            QString outputNodeFullName=edgeiter.key().first;
            uint outputPortID=edgeiter.value().first;
            const OutputPort * outputport=_nodes[outputNodeFullName].second->getOutputPort(outputPortID);
            disconnect(outputport,OUTPUTPORT_SIGNAL
                    ,inputport,INPUTPORT_SLOT);
            _edges.remove(edgeiter.key(),edgeiter.value());
            edgeiter--;
        }
    }
}

void Graph::clearEdges()
{
    QMultiMap< QPair< QString, QString >, QPair< uint, uint > >::const_iterator edgeiter;
    for(edgeiter=_edges.begin();edgeiter!=_edges.end();edgeiter++)
    {
        QString outputNodeFullName=edgeiter.key().first;
        uint outputPortID=edgeiter.value().first;
        const OutputPort * outputport=_nodes[outputNodeFullName].second->getOutputPort(outputPortID);
        QString inputNodeFullName=edgeiter.key().second;
        uint inputPortID=edgeiter.value().second;
        const InputPort * inputport=_nodes[inputNodeFullName].second->getInputPort(inputPortID);
        disconnect(outputport,SIGNAL(signalSendParamsData(TRANSFER_PORT_PARAMS_TYPE,TRANSFER_PORT_DATA_TYPE))
                ,inputport,SLOT(slotReceiveParamsData(TRANSFER_PORT_PARAMS_TYPE,TRANSFER_PORT_DATA_TYPE)));
    }
    _edges.clear();
}

void Graph::changeNodeExName(QString oldNodeFullName, QString newNodeFullName)
{
    if(!_nodes.contains(oldNodeFullName))
    {
        qDebug()<<QString("%1 does not exist in the graph.").arg(oldNodeFullName);
        emit changeNodeExNameResult(0,oldNodeFullName,NULL);
        return;
    }
    if(_nodes.contains(newNodeFullName))
    {
        qDebug()<<QString("%1 has already existed in the graph.").arg(newNodeFullName);
        emit changeNodeExNameResult(0,oldNodeFullName,NULL);
        return;
    }
    QStringList oldnamelist=oldNodeFullName.split(QString("::"),QString::SkipEmptyParts);
    QStringList newnamelist=newNodeFullName.split(QString("::"),QString::SkipEmptyParts);
    if(newnamelist.size()<2||newnamelist.size()>3)
    {
        emit changeNodeExNameResult(0,oldNodeFullName,NULL);
        return;
    }
    else if(oldnamelist.at(0)!=newnamelist.at(0)||oldnamelist.at(1)!=newnamelist.at(1))
    {
        emit changeNodeExNameResult(0,oldNodeFullName,NULL);
        return;
    }

    QMultiMap< QPair< QString, QString >, QPair< uint, uint > > oldedgesback;
    QMultiMap< QPair< QString, QString >, QPair< uint, uint > > newedgesback;
    QMultiMap< QPair< QString, QString >, QPair< uint, uint > >::const_iterator edgeiter;
    for(edgeiter=_edges.begin();edgeiter!=_edges.end();edgeiter++)
    {
        if(edgeiter.key().first==oldNodeFullName)
        {
            oldedgesback.insert(edgeiter.key(),edgeiter.value());
            if(edgeiter.key().second==oldNodeFullName)
            {
                newedgesback.insert(QPair< QString, QString >(newNodeFullName,newNodeFullName),edgeiter.value());
            }
            else
            {
                newedgesback.insert(QPair< QString, QString >(newNodeFullName,edgeiter.key().second),edgeiter.value());
            }
        }
        else if(edgeiter.key().second==oldNodeFullName)
        {
            oldedgesback.insert(edgeiter.key(),edgeiter.value());
            newedgesback.insert(QPair< QString, QString >(edgeiter.key().first,newNodeFullName),edgeiter.value());
        }
    }
    QString libraryfilenameback=_nodes[oldNodeFullName].second->_libraryfilename;
    QString configfilenameback=_nodes[oldNodeFullName].second->_configfilename;
    removeNode(oldNodeFullName);
    addNode(newNodeFullName,libraryfilenameback,configfilenameback);
    if(_nodes.contains(newNodeFullName))
    {        
        emit changeNodeExNameResult(1,oldNodeFullName,_nodes[oldNodeFullName].second);
        for(edgeiter=newedgesback.begin();edgeiter!=newedgesback.end();edgeiter++)
        {
            addEdge(edgeiter.key().first,edgeiter.value().first,edgeiter.key().second,edgeiter.value().second);
        }
    }
    else
    {
        addNode(oldNodeFullName,libraryfilenameback,configfilenameback);
        emit changeNodeExNameResult(0,oldNodeFullName,_nodes[oldNodeFullName].second);
        for(edgeiter=oldedgesback.begin();edgeiter!=oldedgesback.end();edgeiter++)
        {
            addEdge(edgeiter.key().first,edgeiter.value().first,edgeiter.key().second,edgeiter.value().second);
        }        
    }
}

void Graph::changeNodeLibrary(QString nodeFullName, QString libraryFileName)
{
    if(!_nodes.contains(nodeFullName))
    {
        qDebug()<<QString("%1 does not exist in the graph.").arg(nodeFullName);
        emit changeNodeLibraryResult(0,nodeFullName,NULL);
        return;
    }
    QMultiMap< QPair< QString, QString >, QPair< uint, uint > > edgesback;
    QMultiMap< QPair< QString, QString >, QPair< uint, uint > >::const_iterator edgeiter;
    for(edgeiter=_edges.begin();edgeiter!=_edges.end();edgeiter++)
    {
        if(edgeiter.key().first==nodeFullName||edgeiter.key().second==nodeFullName)
        {
            edgesback.insert(edgeiter.key(),edgeiter.value());
        }
    }
    QString libraryfilenameback=_nodes[nodeFullName].second->_libraryfilename;
    QString configfilenameback=_nodes[nodeFullName].second->_configfilename;
    removeNode(nodeFullName);
    addNode(nodeFullName,libraryFileName,configfilenameback);
    if(_nodes.contains(nodeFullName))
    {
        emit changeNodeLibraryResult(1,nodeFullName,_nodes[nodeFullName].second);
    }
    else
    {
        addNode(nodeFullName,libraryfilenameback,configfilenameback);
        emit changeNodeLibraryResult(0,nodeFullName,_nodes[nodeFullName].second);
    }
    for(edgeiter=edgesback.begin();edgeiter!=edgesback.end();edgeiter++)
    {
        addEdge(edgeiter.key().first,edgeiter.value().first,edgeiter.key().second,edgeiter.value().second);
    }
}

void Graph::changeNodeConfigFile(QString nodeFullName, QString configFileName)
{
    if(!_nodes.contains(nodeFullName))
    {
        qDebug()<<QString("%1 does not exist in the graph.").arg(nodeFullName);
        return;
    }
    _nodes[nodeFullName].second->_libraryfilename=configFileName;
}

void Graph::openNode(QString nodeFullName)
{
    if(!_nodes.contains(nodeFullName))
    {
        qDebug()<<QString("%1 does not exist in the graph.").arg(nodeFullName);
        return;
    }
    QEvent * event=new QEvent(QEvent::Type(OpenNodeEventType));
    QCoreApplication::postEvent(_nodes[nodeFullName].second,event);
}

void Graph::closeNode(QString nodeFullName)
{
    if(!_nodes.contains(nodeFullName))
    {
        qDebug()<<QString("%1 does not exist in the graph.").arg(nodeFullName);
        return;
    }
    QEvent * event=new QEvent(QEvent::Type(CloseNodeEventType));
    QCoreApplication::postEvent(_nodes[nodeFullName].second,event);
}

void Graph::openAllNode()
{
    QMap< QString, QPair< std::shared_ptr< QThread >, Node * > >::const_iterator nodeiter;
    for(nodeiter=_nodes.begin();nodeiter!=_nodes.end();nodeiter++)
    {
        QEvent * event=new QEvent(QEvent::Type(OpenNodeEventType));
        QCoreApplication::postEvent(nodeiter.value().second,event);
    }
}

void Graph::closeAllNode()
{
    QMap< QString, QPair< std::shared_ptr< QThread >, Node * > >::const_iterator nodeiter;
    for(nodeiter=_nodes.begin();nodeiter!=_nodes.end();nodeiter++)
    {
        QEvent * event=new QEvent(QEvent::Type(CloseNodeEventType));
        QCoreApplication::postEvent(nodeiter.value().second,event);
    }
}

void Graph::showWidget(QString nodeFullName)
{
    if(!_nodes.contains(nodeFullName))
    {
        qDebug()<<QString("%1 does not exist in the graph.").arg(nodeFullName);
        return;
    }
    if(_nodes[nodeFullName].second->NODE_VARS_ARG->_guithreadflag
            ||_nodes[nodeFullName].second->NODE_VARS_ARG->_showwidgetflag)
    {
        if(!_nodes[nodeFullName].second->NODE_VARS_ARG->getWidget()->isVisible())
        {
            _nodes[nodeFullName].second->NODE_VARS_ARG->getWidget()->setVisible(1);
            if(widgetgeometry.contains(nodeFullName))
            {
                _nodes[nodeFullName].second->NODE_VARS_ARG->getWidget()->setGeometry(widgetgeometry[nodeFullName]);
            }
            else
            {
                widgetgeometry.insert(nodeFullName,_nodes[nodeFullName].second->NODE_VARS_ARG->getWidget()->geometry());
            }
        }
    }
}

void Graph::hideWidget(QString nodeFullName)
{
    if(!_nodes.contains(nodeFullName))
    {
        qDebug()<<QString("%1 does not exist in the graph.").arg(nodeFullName);
        return;
    }
    if(_nodes[nodeFullName].second->NODE_VARS_ARG->_guithreadflag
            ||_nodes[nodeFullName].second->NODE_VARS_ARG->_showwidgetflag)
    {
        if(_nodes[nodeFullName].second->NODE_VARS_ARG->getWidget()->isVisible())
        {
            _nodes[nodeFullName].second->NODE_VARS_ARG->getWidget()->setVisible(0);
            widgetgeometry[nodeFullName]=_nodes[nodeFullName].second->NODE_VARS_ARG->getWidget()->geometry();
        }
    }
}

void Graph::showAllWidget()
{
    QStringList nodelist=_nodes.keys();
    uint i,n=nodelist.size();
    for(i=0;i<n;i++)
    {
        showWidget(nodelist.at(i));
    }
}

void Graph::hideAllWidget()
{
    QStringList nodelist=_nodes.keys();
    uint i,n=nodelist.size();
    for(i=0;i<n;i++)
    {
        hideWidget(nodelist.at(i));
    }
}

const Node *Graph::getNode(QString nodeFullName)
{
    if(!_nodes.contains(nodeFullName))
    {
        qDebug()<<QString("%1 does not exist in the graph.").arg(nodeFullName);
        return NULL;
    }
    return _nodes[nodeFullName].second;
}

const QWidget *Graph::getNodeWidget(QString nodeFullName)
{
    if(!_nodes.contains(nodeFullName))
    {
        qDebug()<<QString("%1 does not exist in the graph.").arg(nodeFullName);
        return NULL;
    }
    return _nodes[nodeFullName].second->NODE_VARS_ARG->getWidget();
}

bool Graph::contains(QString nodeFullName)
{
    return _nodes.contains(nodeFullName);
}
