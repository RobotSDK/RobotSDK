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
#include<QMenu>
#include<QAction>
#include<QGraphicsSceneResizeEvent>

#include<assert.h>

#include"xport.h"

#include<RobotSDK_Global.h>

class XNode : public QGraphicsProxyWidget
{
    Q_OBJECT
    friend class XGraph;
public:
    XNode(RobotSDK::Graph * graph, QString nodeFullName);
    ~XNode();
protected:
    RobotSDK::Graph * _graph;
    const RobotSDK::Node * _node;
    uint _inputportnum;
    uint _outputportnum;
    QString tmpnewnodefullname;
public:
    bool resizeFlag=0;
    QWidget * widget=NULL;
    QLabel * nodefullname=NULL;
    QPushButton * changeexname=NULL;
    QLineEdit * libraryfilename=NULL;
    QPushButton * changelibraryfilename=NULL;
    QLineEdit * configfilename=NULL;
    QPushButton * changeconfigfilename=NULL;
    QPushButton * opennode=NULL;
    QPushButton * showwidget=NULL;
    QPushButton * generatecode=NULL;
protected:
    QList<XPort *> inputportslist;
    QList<XPort *> outputportslist;
protected:
    void resizeEvent(QGraphicsSceneResizeEvent * event);
signals:
    void signalResize(QString nodeFullName, QSizeF newSize);
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
    void slotAddEdge(QString outputNodeFullName, uint outputPortID, QString inputNodeFullName, uint inputPortID);
signals:
    void signalAddEdge(QString outputNodeFullName, uint outputPortID, QString inputNodeFullName, uint inputPortID);
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
    void slotAddNodeResult(bool successFlag, QString nodeFullName, const RobotSDK::Node * node);
    void slotChangeNodeExNameResult(bool successFlag, QString oldNodeFullName, const RobotSDK::Node * node);
    void slotChangeNodeLibraryResult(bool successFlag, QString nodeFullName, const RobotSDK::Node * node);
signals:
    void signalUpdateNode(QString oldNodeFullName, QString newNodeFullName);
protected slots:
    void slotNodeFullNameMenu(const QPoint & pos);
signals:
    void signalRemoveNode(QString nodeFullName);
protected slots:
    void slotGenerateCode();
};

#endif // XNODE_H
