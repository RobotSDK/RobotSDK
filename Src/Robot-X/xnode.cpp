#include "xnode.h"

XNode::XNode(RobotSDK::Graph *graph, QString nodeFullName)
    : QGraphicsProxyWidget(NULL)
{
    assert(graph!=NULL);

    _graph=graph;
    connect(this,SIGNAL(signalOpenNode(QString)),_graph,SLOT(openNode(QString)));
    connect(this,SIGNAL(signalCloseNode(QString)),_graph,SLOT(closeNode(QString)));
    connect(this,SIGNAL(signalShowWidget(QString)),_graph,SLOT(showWidget(QString)));
    connect(this,SIGNAL(signalHideWidget(QString)),_graph,SLOT(hideWidget(QString)));

    connect(this,SIGNAL(signalAddEdge(QString,uint,QString,uint)),_graph,SLOT(addEdge(QString,uint,QString,uint)));
    connect(this,SIGNAL(signalChangeNodeExName(QString,QString)),_graph,SLOT(changeNodeExName(QString,QString)));
    connect(this,SIGNAL(signalChangeNodeLibrary(QString,QString)),_graph,SLOT(changeNodeLibrary(QString,QString)));
    connect(this,SIGNAL(signalAddEdge(QString,uint,QString,uint)),_graph,SLOT(addNode(QString,QString,QString)));
    connect(this,SIGNAL(signalChangeNodeConfigFile(QString,QString)),_graph,SLOT(changeNodeConfigFile(QString,QString)));
    connect(_graph,SIGNAL(changeNodeResult(bool, const RobotSDK::Node *)),this,SLOT(slotChangeNodeResult(bool,const RobotSDK::Node*)));

    uint i;
    QLabel *label;
    QPalette palette;

    QWidget * widget=new QWidget;
    this->setWidget(widget);

    QHBoxLayout * layout=new QHBoxLayout;
    widget->setLayout(layout);

    QVBoxLayout * inputports=new QVBoxLayout;
    layout->addLayout(inputports);

    QVBoxLayout * nodelayout=new QVBoxLayout;
    layout->addLayout(nodelayout);

    nodefullname=new QLabel(nodeFullName);
    nodefullname->setAlignment(Qt::AlignCenter);
    nodelayout->addWidget(nodefullname);

    changeexname=new QPushButton("Change ExName");
    connect(changeexname,SIGNAL(clicked()),this,SLOT(slotChangeNodeExName()));
    nodelayout->addWidget(changeexname);

    libraryfilename=new QLineEdit;
    libraryfilename->setReadOnly(1);
    nodelayout->addWidget(libraryfilename);

    changelibraryfilename=new QPushButton("Change Library");
    connect(changelibraryfilename,SIGNAL(clicked()),this,SLOT(slotChangeNodeLibrary()));
    nodelayout->addWidget(changelibraryfilename);

    configfilename=new QLineEdit;
    configfilename->setReadOnly(1);
    nodelayout->addWidget(configfilename);

    changeconfigfilename=new QPushButton("Change Configure");
    connect(changeconfigfilename,SIGNAL(clicked()),this,SLOT(slotChangeNodeConfigFile()));
    nodelayout->addWidget(changeconfigfilename);

    opennode=new QPushButton("Open Node");
    connect(opennode,SIGNAL(clicked()),this,SLOT(slotOpenNode()));
    nodelayout->addWidget(opennode);
    palette=opennode->palette();
    palette.setColor(QPalette::Button, QColor(Qt::red));
    opennode->setPalette(palette);

    showwidget=new QPushButton("Show Widget");
    connect(showwidget,SIGNAL(clicked()),this,SLOT(slotShowWidget()));
    nodelayout->addWidget(showwidget);
    palette=showwidget->palette();
    palette.setColor(QPalette::Button, QColor(Qt::red));
    showwidget->setPalette(palette);

    generatecode=new QPushButton("Generate Code");
    connect(generatecode,SIGNAL(clicked()),this,SLOT(slotGenerateCode()));
    nodelayout->addWidget(generatecode);

    QVBoxLayout * outputports=new QVBoxLayout;
    layout->addLayout(outputports);

    uint inputportnum;
    uint outputportnum;

    if(graph->contains(nodeFullName))
    {
        _node=_graph->getNode(nodeFullName);
        connect(_node,SIGNAL(signalNodeState(bool,QString)),this,SLOT(slotNodeState(bool,QString)),Qt::QueuedConnection);

        inputportnum=_node->_inputportnum;
        outputportnum=_node->_outputportnum;

        libraryfilename->setText(_node->_libraryfilename);
        configfilename->setText(_node->_configfilename);
    }
    else
    {
        _node=NULL;

        inputportnum=QInputDialog::getInt(widget,"Set Number of Input Ports","Number of Input Ports",0);
        outputportnum=QInputDialog::getInt(widget,"Set Number of Output Ports","Number of Output Ports",0);

        libraryfilename->setText("Developing...");
        configfilename->setText("Config.xml");
    }
    label=new QLabel(QString("%1 Input Ports").arg(inputportnum));
    label->setFrameStyle(QFrame::Panel|QFrame::Sunken);
    label->setAlignment(Qt::AlignCenter);
    inputports->addWidget(label);

    inputportslist.clear();
    for(i=0;i<inputportnum;i++)
    {
        XPort * port=new XPort;
        port->setFrameStyle(QFrame::Panel|QFrame::Sunken);
        port->setText(QString("#%1").arg(i));
        port->setAlignment(Qt::AlignCenter);
        port->porttype=XPort::InputPort;
        port->nodefullname=nodeFullName;
        port->portid=i;
        inputports->addWidget(port);
        connect(port,SIGNAL(signalAddEdge(QString,uint,QString,uint)),this,SLOT(slotAddEdge(QString,uint,QString,uint)));
        inputportslist.push_back(port);
    }

    label=new QLabel(QString("%1 Output Port").arg(outputportnum));
    label->setFrameStyle(QFrame::Panel|QFrame::Sunken);
    label->setAlignment(Qt::AlignCenter);
    outputports->addWidget(label);

    outputportslist.clear();
    for(i=0;i<outputportnum;i++)
    {
        XPort * port=new XPort;
        port->setFrameStyle(QFrame::Panel|QFrame::Sunken);
        port->setText(QString("#%1").arg(i));
        port->setAlignment(Qt::AlignCenter);
        port->porttype=XPort::OutputPort;
        port->nodefullname=nodeFullName;
        port->portid=i;
        outputports->addWidget(port);
        connect(port,SIGNAL(signalAddEdge(QString,uint,QString,uint)),this,SLOT(slotAddEdge(QString,uint,QString,uint)));
        outputportslist.push_back(port);
    }
}

XNode::~XNode()
{

}

void XNode::slotOpenNode()
{
    if(_graph->contains(nodefullname->text()))
    {
        if(opennode->text()==QString("Open Node"))
        {
            emit signalOpenNode(nodefullname->text());
        }
        else
        {
            emit signalCloseNode(nodefullname->text());
        }
    }
}

void XNode::slotNodeState(bool openFlag, QString nodeFullName)
{
    Q_UNUSED(nodeFullName);
    if(openFlag)
    {
        QPalette palette=opennode->palette();
        palette.setColor(QPalette::Button, QColor(Qt::green));
        opennode->setPalette(palette);
        opennode->setText("Close Node");
    }
    else
    {
        QPalette palette=opennode->palette();
        palette.setColor(QPalette::Button, QColor(Qt::red));
        opennode->setPalette(palette);
        opennode->setText("Open Node");
    }
}

void XNode::slotShowWidget()
{
    if(_graph->contains(nodefullname->text()))
    {
        if(showwidget->text()==QString("Show Widget"))
        {
            emit signalShowWidget(nodefullname->text());
            QPalette palette=showwidget->palette();
            palette.setColor(QPalette::Button, QColor(Qt::green));
            showwidget->setPalette(palette);
            showwidget->setText("Hide Widget");
        }
        else
        {
            emit signalHideWidget(nodefullname->text());
            QPalette palette=showwidget->palette();
            palette.setColor(QPalette::Button, QColor(Qt::red));
            showwidget->setPalette(palette);
            showwidget->setText("Show Widget");
        }
    }
}

void XNode::slotAddEdge(QString outputNodeFullName, uint outputPortID, QString inputNodeFull, uint inputPortID)
{
    emit signalAddEdge(outputNodeFullName,outputPortID,inputNodeFull,inputPortID);
}

void XNode::slotChangeNodeExName()
{
    QString newnodefullname=QInputDialog::getText(changeexname,"Input Node Full Name with New ExName","New Node Full Name",QLineEdit::Normal,nodefullname->text());
    if(newnodefullname.size()>0)
    {
        if(_graph->contains(nodefullname->text()))
        {
            emit signalChangeNodeExName(nodefullname->text(),newnodefullname);
        }
        else
        {
            QStringList oldnodefullnamelist=nodefullname->text().split(QString("::"),QString::SkipEmptyParts);
            QStringList newnodefullnamelist=newnodefullname.split(QString("::"),QString::SkipEmptyParts);
            if(newnodefullnamelist.size()>1&&newnodefullnamelist.size()<4)
            {
                if(oldnodefullnamelist.at(0)==newnodefullnamelist.at(0)&&oldnodefullnamelist.at(1)==newnodefullnamelist.at(1))
                {
                    emit signalNodeUpdate(nodefullname->text(),newnodefullname);
                    nodefullname->setText(newnodefullname);
                }
            }
        }
    }
}

void XNode::slotChangeNodeLibrary()
{
#ifdef Q_OS_LINUX
    QString newlibrary=QFileDialog::getOpenFileName(changelibraryfilename,"Open Library",QString(),QString("Shared Library (*.so)"));
#endif
#ifdef Q_OS_WIN32
    QString newlibrary=QFileDialog::getOpenFileName(this,"Open Library",QString(),QString("Shared Library (*.dll)"));
#endif
    if(newlibrary.size()>0)
    {
        if(_graph->contains(nodefullname->text()))
        {
            emit signalChangeNodeLibrary(nodefullname->text(),newlibrary);
        }
        else
        {
            emit signalAddNode(nodefullname->text(),newlibrary,configfilename->text());
        }
    }
}

void XNode::slotChangeNodeConfigFile()
{
    QString newconfigfile=QFileDialog::getOpenFileName(changeconfigfilename,"Open Library",QString(),QString("XML File (*.xml)"));
    if(newconfigfile.size()>0)
    {
        if(_graph->contains(nodefullname->text()))
        {
            emit signalChangeNodeConfigFile(nodefullname->text(),newconfigfile);
        }
        else
        {
            configfilename->setText(newconfigfile);
        }
    }
}

void XNode::slotChangeNodeResult(bool successFlag, const RobotSDK::Node *node)
{
    connect(node,SIGNAL(signalNodeState(bool,QString)),this,SLOT(slotNodeState(bool,QString)),Qt::QueuedConnection);
    if(successFlag)
    {
        _node=node;
        connect(_node,SIGNAL(signalNodeState(bool,QString)),this,SLOT(slotNodeState(bool,QString)),Qt::QueuedConnection);
        if(nodefullname->text()!=node->_nodefullname)
        {
            emit signalNodeUpdate(nodefullname->text(),node->_nodefullname);
        }
        nodefullname->setText(node->_nodefullname);
        libraryfilename->setText(node->_libraryfilename);
        configfilename->setText(node->_configfilename);
        uint i;
        for(i=0;i<node->_inputportnum;i++)
        {
            inputportslist.at(i)->nodefullname=node->_nodefullname;
        }
        for(i=0;i<node->_outputportnum;i++)
        {
            outputportslist.at(i)->nodefullname=node->_nodefullname;
        }
    }
}

void XNode::slotGenerateCode()
{

}
