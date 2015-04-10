#include "graph.h"

using namespace RobotSDK;

Graph::Graph(QObject * parent)
    : QObject(parent)
{    
    registerTransferData();

    nodeswitcher=new QVBoxLayout;
    widgetswitcher=new QVBoxLayout;
    QHBoxLayout * layout=new QHBoxLayout;
    layout->addLayout(nodeswitcher);
    layout->addLayout(widgetswitcher);
    QVBoxLayout * nestlayout=new QVBoxLayout;
    nestlayout->addLayout(layout);
    nestlayout->addStretch();
    graph->setLayout(nestlayout);
}

Graph::~Graph()
{
    clearNodes();
    delete graph;
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
    if(_nodes.values(nodeFullName).size()>0)
    {
        qDebug()<<QString("%1 has already existed in the graph.").arg(nodeFullName);
        return;
    }
    QStringList nodenamelist=nodeFullName.split(QString("::"),QString::SkipEmptyParts);
    if(nodenamelist.size()<2||nodenamelist.size()>3)
    {
        qDebug()<<QString("%1 is not a valid node full name (NodeClass::NodeName[::ExName]).");
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
    generateNode=(generateNodePtr)(QLibrary::resolve(libraryFileName,functionname.toUtf8().data()));
    if(generateNode==NULL)
    {
        qDebug()<<QString("Can not resolve %1 from %2. May lack of USE_DEFAULT_NODE or USE_EXTENDED_NODE in module source code.").arg(functionname).arg(libraryFileName);
        return;
    }
    Node * node=generateNode(libraryFileName, configFileName, nodeClass, nodeName, exName);
    if(node==NULL)
    {
        qDebug()<<QString("Can not build node. May the node's type is not extended from RobotSDK::Node");
        return;
    }
    if(!node->_loadflag||!node->_initializeflag)
    {
        delete node;
        return;
    }
    std::shared_ptr<QThread> thread=std::shared_ptr<QThread>(new QThread);
    connect(thread.get(),SIGNAL(finished()),node,SLOT(deleteLater()));
    _nodes.insert(nodeFullName,QPair< std::shared_ptr< QThread >, Node * >(thread,node));
    node->moveToThread(thread.get());
    thread->start();

    nodeswitcher->addWidget(node->NODE_VARS_ARG->getNodeSwitcher());
    widgetswitcher->addWidget(node->NODE_VARS_ARG->getWidgetSwitcher());

    return;
}

void Graph::removeNode(QString nodeFullName)
{
    if(_nodes.values(nodeFullName).size()==0)
    {
        qDebug()<<QString("%1 does not exist in the graph.").arg(nodeFullName);
        return;
    }

    QMap< QString, QPair< std::shared_ptr< QThread >, Node * > >::const_iterator nodeiter;
    for(nodeiter=_nodes.begin();nodeiter!=_nodes.end();nodeiter++)
    {
        _edges.remove(QPair< QString, QString >(nodeFullName,nodeiter.key()));
        _edges.remove(QPair< QString, QString >(nodeiter.key(),nodeFullName));
    }

    nodeswitcher->addWidget(nodeiter.value().second->NODE_VARS_ARG->getNodeSwitcher());
    widgetswitcher->addWidget(nodeiter.value().second->NODE_VARS_ARG->getWidgetSwitcher());

    _nodes[nodeFullName].first->quit();
    _nodes[nodeFullName].first->wait();
    _nodes.remove(nodeFullName);
    return;
}

void Graph::clearNodes()
{
    QMap< QString, QPair< std::shared_ptr<QThread>, Node * > >::const_iterator nodeiter;
    for(nodeiter=_nodes.begin();nodeiter!=_nodes.end();nodeiter++)
    {
        nodeswitcher->removeWidget(nodeiter.value().second->NODE_VARS_ARG->getNodeSwitcher());
        widgetswitcher->removeWidget(nodeiter.value().second->NODE_VARS_ARG->getWidgetSwitcher());
        nodeiter.value().first->exit();
    }
    for(nodeiter=_nodes.begin();nodeiter!=_nodes.end();nodeiter++)
    {
        nodeiter.value().first->wait();
    }
    _edges.clear();

}

void Graph::addEdge(QString outputNodeFullName, uint outputPortID, QString inputNodeFullName, uint inputPortID)
{
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
        OutputPort * outputport=_nodes[outputNodeFullName].second->getOutputPort(outputPortID);
        InputPort * inputport=_nodes[inputNodeFullName].second->getInputPort(inputPortID);
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
        OutputPort * outputport=_nodes[outputNodeFullName].second->getOutputPort(outputPortID);
        InputPort * inputport=_nodes[inputNodeFullName].second->getInputPort(inputPortID);
        disconnect(outputport,OUTPUTPORT_SIGNAL
                ,inputport,INPUTPORT_SLOT);
    }
}

void Graph::removeEdgeByOutputPort(QString outputNodeFullName, uint outputPortID)
{
    if(_nodes.values(outputNodeFullName).size()==0)
    {
        qDebug()<<QString("%1 does not exist in the graph.").arg(outputNodeFullName);
        return;
    }
    OutputPort * outputport=_nodes[outputNodeFullName].second->getOutputPort(outputPortID);
    QMultiMap< QPair< QString, QString >, QPair< uint, uint > >::const_iterator edgeiter;
    for(edgeiter=_edges.begin();edgeiter!=_edges.end();edgeiter++)
    {
        if(edgeiter.key().first==outputNodeFullName)
        {
            QString inputNodeFullName=edgeiter.key().second;
            uint inputPortID=edgeiter.value().second;
            InputPort * inputport=_nodes[inputNodeFullName].second->getInputPort(inputPortID);
            disconnect(outputport,OUTPUTPORT_SIGNAL
                    ,inputport,INPUTPORT_SLOT);
            _edges.remove(edgeiter.key(),edgeiter.value());
            edgeiter--;
        }
    }
}

void Graph::removeEdgeByInputPort(QString inputNodeFullName, uint inputPortID)
{
    if(_nodes.values(inputNodeFullName).size()==0)
    {
        qDebug()<<QString("%1 does not exist in the graph.").arg(inputNodeFullName);
        return;
    }
    InputPort * inputport=_nodes[inputNodeFullName].second->getInputPort(inputPortID);
    QMultiMap< QPair< QString, QString >, QPair< uint, uint > >::const_iterator edgeiter;
    for(edgeiter=_edges.begin();edgeiter!=_edges.end();edgeiter++)
    {
        if(edgeiter.key().second==inputNodeFullName)
        {
            QString outputNodeFullName=edgeiter.key().first;
            uint outputPortID=edgeiter.value().first;
            OutputPort * outputport=_nodes[outputNodeFullName].second->getOutputPort(outputPortID);
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
        OutputPort * outputport=_nodes[outputNodeFullName].second->getOutputPort(outputPortID);
        QString inputNodeFullName=edgeiter.key().second;
        uint inputPortID=edgeiter.value().second;
        InputPort * inputport=_nodes[inputNodeFullName].second->getInputPort(inputPortID);
        disconnect(outputport,SIGNAL(signalSendParamsData(TRANSFER_PORT_PARAMS_TYPE,TRANSFER_PORT_DATA_TYPE))
                ,inputport,SLOT(slotReceiveParamsData(TRANSFER_PORT_PARAMS_TYPE,TRANSFER_PORT_DATA_TYPE)));
    }
    _edges.clear();
}

void Graph::openAllNode()
{
    QMap< QString, QPair< std::shared_ptr< QThread >, Node * > >::const_iterator nodeiter;
    for(nodeiter=_nodes.begin();nodeiter!=_nodes.end();nodeiter++)
    {
        QEvent * openevent=new QEvent(QEvent::Type(NodeSwitcher::OpenNodeEventType));
        QCoreApplication::postEvent(nodeiter.value().second,openevent);
    }
}

void Graph::closeAllNode()
{
    QMap< QString, QPair< std::shared_ptr< QThread >, Node * > >::const_iterator nodeiter;
    for(nodeiter=_nodes.begin();nodeiter!=_nodes.end();nodeiter++)
    {
        QEvent * openevent=new QEvent(QEvent::Type(NodeSwitcher::CloseNodeEventType));
        QCoreApplication::postEvent(nodeiter.value().second,openevent);
    }
}

QWidget * Graph::getNodeWidget(QString nodeFullName)
{
    if(_nodes.values(nodeFullName).size()==0)
    {
        qDebug()<<QString("%1 does not exist in the graph.").arg(nodeFullName);
        return NULL;
    }
    return _nodes[nodeFullName].second->NODE_VARS_ARG->getWidget();
}
