#ifndef XPORT_H
#define XPORT_H

#include<QLabel>
#include<QDrag>
#include<QMimeData>
#include<QPixmap>
#include<QPainter>
#include<QDragEnterEvent>
#include<QDragLeaveEvent>
#include<QDropEvent>
#include<QStringList>
#include<QFontMetrics>
#include<QInputDialog>
#include<QMenu>
#include<QAction>
#include<QBitmap>

#include<RobotSDK.h>

class XPortHead : public QLabel
{
    Q_OBJECT
public:
    XPortHead(QWidget * parent=0);
    XPortHead(QString text, QWidget * parent=0);
public:
    QString nodefullname;
    uint portnum;
protected:
    void mousePressEvent(QMouseEvent * event);
signals:
    void signalResetPortNum(QString text, uint portNum);
};

class XPort : public QLabel
{
    Q_OBJECT
public:
    XPort(QWidget * parent =0);
    ~XPort();
public:
    enum PORTTYPE
    {
        InputPort,
        OutputPort
    };
public:
    PORTTYPE porttype;
    QString nodefullname;
    uint portid;
signals:
    void signalAddEdge(QString outputNodeFullName, uint outputPortID, QString inputNodeFull, uint inputPortID);
    void signalRemovePort(XPort::PORTTYPE portType, QString nodeFullName,uint portID);
protected:
    void mousePressEvent(QMouseEvent *event);
    void dragEnterEvent(QDragEnterEvent * event);
    void dragLeaveEvent(QDragLeaveEvent * event);
    void dragMoveEvent(QDragMoveEvent * event);
    void dropEvent(QDropEvent * event);
};

#endif // XPORT_H
