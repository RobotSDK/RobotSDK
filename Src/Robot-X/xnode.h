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
#include<QFile>
#include<QTextStream>
#include<QDir>
#include<QFileInfo>
#include<QTableWidget>
#include<QDialog>
#include<QComboBox>

#include<xmldominterface.h>
#include<assert.h>

#include"xport.h"

using namespace RobotSDK;

namespace RobotX
{

class XNode;

class XConfigPanel : public QDialog
{
    Q_OBJECT
public:
    XConfigPanel(QString nodeFullName, QString configFilaName, XNode * nodeParent, QWidget * parent=0);
public:
    QString nodefullname;
    QString configfilename;
    QTableWidget * table;
    QVector<bool> optionflag;
    XNode * node;
protected:
    void load();
public slots:
    void accept();
    void apply();
    void reject();
public:
    void showPanel();
};

class XNode : public QGraphicsProxyWidget
{
    Q_OBJECT
    friend class XGraph;
public:
    XNode(Graph * graph, QString nodeFullName);
    ~XNode();
public:
    Graph * _graph;
    const Node * _node;
    uint _inputportnum;
    uint _outputportnum;
    QString tmpnewnodefullname;
public:
    bool resizeFlag=0;
    QWidget * widget=NULL;
    QVBoxLayout * inputports=NULL;
    QVBoxLayout * nodelayout=NULL;
    QVBoxLayout * outputports=NULL;
    QLabel * nodefullname=NULL;
    QLineEdit * libraryfilename=NULL;
    QLineEdit * configfilename=NULL;
    QPushButton * opennode=NULL;
    QPushButton * showwidget=NULL;
    XConfigPanel * configpanel=NULL;
protected:
    QList<XPort *> inputportslist;
    QList<QString> inputportclass;
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
public slots:
    void slotAddEdge(QString outputNodeFullName, uint outputPortID, QString inputNodeFullName, uint inputPortID);
    void slotRemovePort(XPort::PORTTYPE portType, QString nodeFullName,uint portID);
    void slotResetPortNum(QString text, uint portNum);
signals:
    void signalAddEdge(QString outputNodeFullName, uint outputPortID, QString inputNodeFullName, uint inputPortID);
    void signalRemovePort(XPort::PORTTYPE portType, QString nodeFullName,uint portID);
    void signalResetPortNum(QString nodeFullName);
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
    void slotAddNodeResult(bool successFlag, QString nodeFullName, const Node * node);
    void slotChangeNodeExNameResult(bool successFlag, QString oldNodeFullName, const Node * node);
    void slotChangeNodeLibraryResult(bool successFlag, QString nodeFullName, const Node * node);
signals:
    void signalUpdateNode(QString oldNodeFullName, QString newNodeFullName);
protected slots:
    void slotNodeFullNameMenu(const QPoint & pos);
signals:
    void signalRemoveNode(QString nodeFullName);
protected slots:
    void slotGenerateCode(QString dir);
};

}

#endif // XNODE_H
