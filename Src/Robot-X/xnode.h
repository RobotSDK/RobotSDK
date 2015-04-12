#ifndef XNODE_H
#define XNODE_H

#include<QGraphicsWidget>
#include<QLayout>
#include<QLabel>
#include<QLineEdit>
#include<QPushButton>
#include<QCoreApplication>
#include<QInputDialog>
#include<QFileDialog>
#include<QGraphicsProxyWidget>

#include<assert.h>

#include"xport.h"

#include<RobotSDK_Global.h>

class XNode : public QGraphicsProxyWidget
{
    Q_OBJECT
public:
    XNode(RobotSDK::Graph * graph, QString nodeFullName);
    ~XNode();
protected:
    RobotSDK::Graph * _graph;
    const RobotSDK::Node * _node;
protected:
    QLabel * nodefullname;
    QPushButton * changeexname;
    QLineEdit * libraryfilename;
    QPushButton * changelibraryfilename;
    QLineEdit * configfilename;
    QPushButton * changeconfigfilename;
    QPushButton * opennode;
    QPushButton * showwidget;
    QPushButton * generatecode;
protected:
    QList<XPort *> inputportslist;
    QList<XPort *> outputportslist;
protected slots:
    void slotOpenNode();
    void slotNodeState(bool openFlag, QString nodeFullName);
    void slotShowWidget();
signals:
    void signalOpenNode(QString nodeFullName);
    void signalCloseNode(QString nodeFullName);
    void signalShowWidget(QString nodeFullName);
    void signalHideWidget(QString nodeFullName);
protected slots:
    void slotAddEdge(QString outputNodeFullName, uint outputPortID, QString inputNodeFull, uint inputPortID);
signals:
    void signalAddEdge(QString outputNodeFullName, uint outputPortID, QString inputNodeFull, uint inputPortID);
protected slots:
    void slotChangeNodeExName();
    void slotChangeNodeLibrary();
    void slotChangeNodeConfigFile();
signals:
    void signalChangeNodeExName(QString oldNodeFullName, QString newNodeFullName);
    void signalChangeNodeLibrary(QString nodeFullName, QString libraryFileName);
    void signalAddNode(QString nodeFullName, QString libraryFileName, QString configFileName);
    void signalChangeNodeConfigFile(QString nodeFullName, QString configFileName);
protected slots:
    void slotChangeNodeResult(bool successFlag, const RobotSDK::Node * node);
signals:
    void signalNodeUpdate(QString oldNodeFullName, QString newNodeFullName);
protected slots:
    void slotGenerateCode();
};

#endif // XNODE_H
