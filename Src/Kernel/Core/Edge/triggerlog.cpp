#include "triggerlog.h"

TriggerLog::TriggerLog(QWidget *parent, Node * node, bool gotoThread)
    : QWidget(parent)
{
    backupnode=node;
    nodetag=QString("%1_%2_%3").arg(node->getNodeType()).arg(node->getNodeClass()).arg(node->getNodeName());
    QVBoxLayout * layout=new QVBoxLayout();
    layout->addWidget(&log);
    QHBoxLayout * hlayout=new QHBoxLayout();
    hlayout->addStretch();
    QPushButton * save=new QPushButton("Save...");
    hlayout->addWidget(save);
    QPushButton * clear=new QPushButton("Clear");
    hlayout->addWidget(clear);
    layout->addLayout(hlayout);
    this->setLayout(layout);

    log.setReadOnly(1);

    bool flag=1;
    flag&=bool(connect(save,SIGNAL(clicked()),this,SLOT(saveSlot())));
    flag&=bool(connect(clear,SIGNAL(clicked()),this,SLOT(clearSlot())));
    flag&=bool(connect(node,SIGNAL(openNodeSignal()),this,SLOT(openNodeSlot())));
    flag&=bool(connect(node,SIGNAL(openNodeErrorSignal()),this,SLOT(openNodeErrorSlot())));
    flag&=bool(connect(node,SIGNAL(closeNodeSignal()),this,SLOT(closeNodeSlot())));
    flag&=bool(connect(node,SIGNAL(closeNodeErrorSignal()),this,SLOT(closeNodeErrorSlot())));
    if(gotoThread)
    {
        flag&=bool(connect(node,SIGNAL(nodeTriggerTimeSignal(QDateTime, Node::NodeTriggerState)),this,SLOT(nodeTriggerTimeSlot(QDateTime, Node::NodeTriggerState)),Qt::BlockingQueuedConnection));
        //flag&=bool(connect(node,SIGNAL(nodeTriggerTimeSignal(QDateTime, Node::NodeTriggerState)),this,SLOT(nodeTriggerTimeSlot(QDateTime, Node::NodeTriggerState))));
    }
    else
    {
        flag&=bool(connect(node,SIGNAL(nodeTriggerTimeSignal(QDateTime, Node::NodeTriggerState)),this,SLOT(nodeTriggerTimeSlot(QDateTime, Node::NodeTriggerState))));
    }
}

TriggerLog::~TriggerLog()
{
    bool flag=1;
    flag&=bool(disconnect(backupnode,SIGNAL(openNodeSignal()),this,SLOT(openNodeSlot())));
    flag&=bool(disconnect(backupnode,SIGNAL(openNodeErrorSignal()),this,SLOT(openNodeErrorSlot())));
    flag&=bool(disconnect(backupnode,SIGNAL(closeNodeSignal()),this,SLOT(closeNodeSlot())));
    flag&=bool(disconnect(backupnode,SIGNAL(closeNodeErrorSignal()),this,SLOT(closeNodeErrorSlot())));
    flag&=bool(disconnect(backupnode,SIGNAL(nodeTriggerTimeSignal(QDateTime, Node::NodeTriggerState)),this,SLOT(nodeTriggerTimeSlot(QDateTime, Node::NodeTriggerState))));
}

void TriggerLog::saveSlot()
{
    QString filename=QFileDialog::getSaveFileName(this,"Save Log");
    if(filename.size()>0)
    {
        QFile file(filename);
        if(file.open(QIODevice::WriteOnly | QIODevice::Text))
        {
            file.write(log.toPlainText().toUtf8());
            file.close();
        }
    }
}

void TriggerLog::clearSlot()
{
    log.clear();
}

void TriggerLog::openNodeSlot()
{
    log.insertPlainText(QString("Open node %1 successfully.\n").arg(nodetag));
}

void TriggerLog::openNodeErrorSlot()
{
    log.insertPlainText(QString("Open node %1 unsuccessfully.\n").arg(nodetag));
}

void TriggerLog::closeNodeSlot()
{
    log.insertPlainText(QString("Close node %1 successfully.\n").arg(nodetag));
}

void TriggerLog::closeNodeErrorSlot()
{
    log.insertPlainText(QString("Close node %1 unsuccessfully.\n").arg(nodetag));
}

void TriggerLog::nodeTriggerTimeSlot(QDateTime curDateTime, Node::NodeTriggerState nodeTriggerState)
{
    switch(nodeTriggerState)
    {
    case Node::NodeTriggerStart:
        log.insertPlainText(QString("%1 : node %2 triggered.\n").arg(curDateTime.toString("yyyy-MM-dd HH:mm:ss:zzz")).arg(nodetag));
        break;
    case Node::NodeTriggerEnd:
        log.insertPlainText(QString("%1 : node %2 ended successfully.\n").arg(curDateTime.toString("yyyy-MM-dd HH:mm:ss:zzz")).arg(nodetag));
        break;
    case Node::NodeTriggerError:
        log.insertPlainText(QString("%1 : node %2 ended unsuccessfully.\n").arg(curDateTime.toString("yyyy-MM-dd HH:mm:ss:zzz")).arg(nodetag));
        break;
    default:
        break;
    }
}
