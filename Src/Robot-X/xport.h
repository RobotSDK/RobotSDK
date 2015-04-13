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

#include<RobotSDK_Global.h>

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
public slots:
    //void slotPopMenu(const QPoint & pos);
protected:
    void mousePressEvent(QMouseEvent *event);
    void dragEnterEvent(QDragEnterEvent * event);
    void dragLeaveEvent(QDragLeaveEvent * event);
    void dragMoveEvent(QDragMoveEvent * event);
    void dropEvent(QDropEvent * event);
};

#endif // XPORT_H
