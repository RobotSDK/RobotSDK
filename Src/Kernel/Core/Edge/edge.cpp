#include "edge.h"

Edge::Edge()
{
    QVBoxLayout * layout=new QVBoxLayout();
    QHBoxLayout * hlayout=new QHBoxLayout();
    hlayout->addWidget(new QLabel("Timer Speed (ms)"));
    hlayout->addWidget(&timerspeed);
    playpause.setText("Play");
    hlayout->addWidget(&playpause);
    hlayout->addStretch();
    hlayout->addWidget(new QLabel("Time Range (ms)"));
    hlayout->addWidget(&timerangeinput);
    hlayout->addWidget(new QLabel("Time Interval (ms)"));
    hlayout->addWidget(&timeintervalinput);
	hlayout->addWidget(new QLabel("Zoom Ratio"));
    hlayout->addWidget(&zoomratioinput);
    QPushButton * set=new QPushButton("Set Time Line");
    hlayout->addWidget(set);

    layout->addLayout(hlayout);

	QWidget * frame=new QWidget;
	frame->setLayout(&panel);	
	QScrollArea * viewarea=new QScrollArea;
	viewarea->setWidget(frame);
    layout->addWidget(viewarea);
    this->setLayout(layout);

    timerspeed.setText("50");
    timerange=1000;
    timeinterval=100;
	zoomratio=1.0;
    timerangeinput.setText(QString("%1").arg(timerange));
    timeintervalinput.setText(QString("%1").arg(timeinterval));
	zoomratioinput.setText(QString("%1").arg(zoomratio));


	panel.addWidget(new QLabel("Trigger Log"),0,0);
	panel.addWidget(new QLabel("Trigger View"),0,1);

    bool flag=1;
    flag&=bool(connect(set,SIGNAL(clicked()),this,SLOT(setTimeLineSlot())));
    flag&=bool(connect(&timer,SIGNAL(timeout()),this,SLOT(drawSlot())));
    flag&=bool(connect(&playpause,SIGNAL(clicked()),this,SLOT(playPauseTimerSlot())));
}

Edge::~Edge()
{
    timer.stop();
    clear();
}

bool Edge::connectNodes(Node * outputNode, Node * inputNode)
{
    QVector<QString> outputnodesname=outputNode->getOutputNodesName();
    QVector<QString> inputnodesname=inputNode->getInputNodesName();
    QString outputnodename=outputNode->getNodeName();
    QString inputnodename=inputNode->getNodeName();
    int i,n;
    n=outputnodesname.size();
    int outputportindex=-1;
    for(i=0;i<n;i++)
    {
        QStringList nodenames=outputnodesname[i].split(";",QString::SkipEmptyParts);
        int j,m=nodenames.size();
        for(j=0;j<m;j++)
        {
            if(nodenames[j]==inputnodename)
            {
                outputportindex=i;
                break;
            }
        }
    }
    n=inputnodesname.size();
    int inputportindex=-1;
    for(i=0;i<n;i++)
    {
        QStringList nodenames=inputnodesname[i].split(";",QString::SkipEmptyParts);
        int j,m=nodenames.size();
        for(j=0;j<m;j++)
        {
            if(nodenames[j]==outputnodename)
            {
                inputportindex=i;
                break;
            }
        }
    }
    if(outputportindex<0||inputportindex<0)
    {
        return 1;
    }
    return bool(connect(outputNode->getOutputPort(outputportindex),SIGNAL(outputDataSignal(boost::shared_ptr<void>, boost::shared_ptr<void>))
        ,inputNode->getInputPort(inputportindex),SLOT(inputDataSlot(boost::shared_ptr<void>,  boost::shared_ptr<void>)),Qt::BlockingQueuedConnection));
}

bool Edge::disconnectNodes(Node * inputNode, Node * outputNode)
{
    QVector<QString> outputnodesname=outputNode->getOutputNodesName();
    QVector<QString> inputnodesname=inputNode->getInputNodesName();
    QString outputnodename=outputNode->getNodeName();
    QString inputnodename=inputNode->getNodeName();
    int i,n;
    n=outputnodesname.size();
    int outputportindex=-1;
    for(i=0;i<n;i++)
    {
        if(outputnodesname[i]==inputnodename)
        {
            outputportindex=i;
            break;
        }
    }
    n=inputnodesname.size();
    int inputportindex=-1;
    for(i=0;i<n;i++)
    {
        if(inputnodesname[i]==outputnodename)
        {
            inputportindex=i;
            break;
        }
    }
    if(outputportindex<0||inputportindex<0)
    {
        return 1;
    }
    return bool(disconnect(outputNode->getOutputPort(outputportindex),SIGNAL(outputDataSignal(boost::shared_ptr<void>, boost::shared_ptr<void>))
        ,inputNode->getInputPort(inputportindex),SLOT(inputDataSlot(boost::shared_ptr<void>,  boost::shared_ptr<void>))));
}

void Edge::addNode(Node * node, bool gotoThread, bool needMonitor)
{
    int n=nodepool.size();
    nodepool.insert(node);
    int m=nodepool.size();
    if(m==n+1)
    {
        bool flag=1;
        if(gotoThread)
        {
            QThread * thread=new QThread();
            node->moveToThread(thread);
            thread->start();
            threads.push_back(thread);
            flag&=bool(connect(this,SIGNAL(openAllNodesSignal()),node,SLOT(openNodeSlot()),Qt::BlockingQueuedConnection));
            //flag&=bool(connect(this,SIGNAL(closeAllNodesSignal()),node,SLOT(closeNodeSlot()),Qt::BlockingQueuedConnection));
            //flag&=bool(connect(this,SIGNAL(openAllNodesSignal()),node,SLOT(openNodeSlot())));
            flag&=bool(connect(this,SIGNAL(closeAllNodesSignal()),node,SLOT(closeNodeSlot())));
        }
        else
        {
            flag&=bool(connect(this,SIGNAL(openAllNodesSignal()),node,SLOT(openNodeSlot())));
            flag&=bool(connect(this,SIGNAL(closeAllNodesSignal()),node,SLOT(closeNodeSlot())));
        }
        if(needMonitor)//&&gotoThread)
        {
            TriggerLog * triggerlog=new TriggerLog(this,node,gotoThread);
            TriggerView * triggerview=new TriggerView(this,node,timerange,timeinterval,zoomratio,gotoThread);
            int row=panel.rowCount();
            //panel.setVerticalHeaderItem(row,new QTableWidgetItem(QString("%1_%2_%3").arg(node->getNodeType()).arg(node->getNodeClass()).arg(node->getNodeName())));
            panel.addWidget(triggerlog,row,0);
            panel.addWidget(triggerview,row,1);
			QWidget * parent=(QWidget *)panel.parent();
			parent->resize(400+int(timerange*zoomratio+0.5),MONITORSIZE*panel.rowCount());
        }
    }
}

void Edge::clear()
{
    bool flag=1;
    int i,n;

    flag&=disconnectAll();
    n=panel.rowCount();
    for(i=n-1;i>=0;i--)
    {
		TriggerLog * triggerlog=(TriggerLog *)(panel.itemAtPosition(i,0)->widget());
        TriggerView * triggerview=(TriggerView *)(panel.itemAtPosition(i,1)->widget());
		panel.removeWidget(triggerlog);
		panel.removeWidget(triggerview);
        delete triggerlog;
        delete triggerview;
    }

    emit closeAllNodesSignal();

    n=threads.size();
    for(i=0;i<n;i++)
    {
        threads[i]->exit();
        threads[i]->wait();
        threads[i]->deleteLater();
    }
    threads.clear();

    QList<Node *> nodes;
    nodes=QList<Node *>::fromSet(nodepool);
    nodepool.clear();
    n=nodes.size();
    for(i=0;i<n;i++)
    {
        flag&=bool(disconnect(this,SIGNAL(openAllNodesSignal()),nodes.at(i),SLOT(openNodeSlot())));
        flag&=bool(disconnect(this,SIGNAL(closeAllNodesSignal()),nodes.at(i),SLOT(closeNodeSlot())));
        nodes[i]->deleteLater();
    }
	QWidget * parent=(QWidget *)panel.parent();
	parent->resize(400+int(timerange*zoomratio+0.5),MONITORSIZE*panel.rowCount());
}

bool Edge::connectAll()
{
    QList<Node *> outputnodes=nodepool.toList();
    QList<Node *> inputnodes=nodepool.toList();
    int i,j,n=nodepool.size();
    bool flag=1;
    for(i=0;i<n;i++)
    {
        for(j=0;j<n;j++)
        {
            flag&=connectNodes(outputnodes[i],inputnodes[j]);
        }
    }
    return flag;
}

bool Edge::disconnectAll()
{
    QList<Node *> outputnodes=nodepool.toList();
    QList<Node *> inputnodes=nodepool.toList();
    int i,j,n=nodepool.size();
    bool flag=1;
    for(i=0;i<n;i++)
    {
        for(j=0;j<n;j++)
        {
            if(i!=j)
            {
                flag&=disconnectNodes(outputnodes[i],inputnodes[j]);
            }
        }
    }
    return flag;
}

void Edge::openAllNodesSlot()
{
    emit openAllNodesSignal();
}

void Edge::closeAllNodesSlot()
{
    emit closeAllNodesSignal();
}

void Edge::playPauseTimerSlot()
{
    if(playpause.text()==QString("Play"))
    {
        int speed=timerspeed.text().toInt();
        if(speed>0)
        {
            playpause.setText("Pause");
            timer.start(speed);
        }
    }
    else
    {
        playpause.setText("Play");
        timer.stop();
    }
}

void Edge::setTimeLineSlot()
{
    int i,n=panel.rowCount();
    timerange=timerangeinput.text().toInt();
    timeinterval=timeintervalinput.text().toInt();
	zoomratio=zoomratioinput.text().toDouble();
    for(i=1;i<n;i++)
    {
        TriggerView * triggerview=(TriggerView *)(panel.itemAtPosition(i,1)->widget());
        triggerview->setTimeLine(timerange,timeinterval,zoomratio);		
    }
	QWidget * parent=(QWidget *)panel.parent();
	parent->resize(400+int(timerange*zoomratio+0.5),MONITORSIZE*panel.rowCount());
}

void Edge::drawSlot()
{
    emit drawSignal(QDateTime::currentDateTime());
}
