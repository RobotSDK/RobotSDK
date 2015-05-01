#include "xport.h"

using namespace RobotX;

XPort::XPort(QWidget * parent)
    :QLabel(parent)
{
    setCursor(Qt::OpenHandCursor);
    this->setStyleSheet("QLabel { background-color : white; color : black; border: 2px solid yellow}");
    this->setAlignment(Qt::AlignCenter);
    setAcceptDrops(1);
}

XPort::~XPort()
{
    emit signalRemovePort(porttype,nodefullname,portid);
}

void XPort::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton)
    {
        setCursor(Qt::ClosedHandCursor);
        QDrag *drag = new QDrag(this);
        QMimeData *mimeData = new QMimeData;
        QString message=QString("%1~~%2~~%3").arg(int(porttype)).arg(nodefullname).arg(portid);
        mimeData->setText(message);
        drag->setMimeData(mimeData);
        QString info;
        switch (porttype)
        {
        case InputPort:
           info=QString("%1\nInput Port_%2").arg(nodefullname).arg(portid);
            break;
        case OutputPort:
            info=QString("%1\nOutput Port_%2").arg(nodefullname).arg(portid);
            break;
        default:
            break;
        }        
        QFontMetrics fm(QApplication::font());
        QPixmap pixmap(QSize(fm.width(info),fm.lineSpacing()*2));
        drag->setHotSpot(QPoint(pixmap.width()/2,pixmap.height()/2));
        pixmap.fill(Qt::white);
        pixmap.setMask(pixmap.createMaskFromColor(QColor(Qt::white)));
        QPainter painter(&pixmap);
        painter.setPen(Qt::black);
        painter.drawText(pixmap.rect(), Qt::AlignCenter, info);
        painter.drawRect(0,0,pixmap.width()-1,pixmap.height()-1);

        drag->setPixmap(pixmap);
        drag->exec();
        setCursor(Qt::OpenHandCursor);
    }
}

void XPort::dragEnterEvent(QDragEnterEvent *event)
{
    if(event->mimeData()->hasText())
    {
        QStringList decode=event->mimeData()->text().split(QString("~~"),QString::SkipEmptyParts);
        if(decode.size()==3)
        {
            if(porttype==PORTTYPE(decode.at(0).toInt()))
            {
                if(decode.at(1)!=nodefullname||decode.at(2).toUInt()!=portid)
                {
                    this->setStyleSheet("QLabel { background-color : red; color : black; border: 2px solid yellow}");
                }
            }
            else
            {
                this->setStyleSheet("QLabel { background-color : green; color : black; border: 2px solid yellow}");
            }
            event->acceptProposedAction();
        }
    }
}

void XPort::dragLeaveEvent(QDragLeaveEvent *event)
{
    Q_UNUSED(event);
    this->setStyleSheet("QLabel { background-color : white; color : black; border: 2px solid yellow}");
}

void XPort::dragMoveEvent(QDragMoveEvent * event)
{
    if(event->mimeData()->hasText())
    {
        QStringList decode=event->mimeData()->text().split(QString("~~"),QString::SkipEmptyParts);
        if(decode.size()==3)
        {
            if(porttype==PORTTYPE(decode.at(0).toInt()))
            {
                if(decode.at(1)!=nodefullname||decode.at(2).toUInt()!=portid)
                {
                    this->setStyleSheet("QLabel { background-color : red; color : black; border: 2px solid yellow}");
                }
            }
            else
            {
                this->setStyleSheet("QLabel { background-color : green; color : black; border: 2px solid yellow}");
            }
            event->acceptProposedAction();
        }
    }
}

void XPort::dropEvent(QDropEvent *event)
{
    if(event->mimeData()->hasText())
    {
        QStringList decode=event->mimeData()->text().split(QString("~~"),QString::SkipEmptyParts);
        if(decode.size()==3&&porttype!=PORTTYPE(decode.at(0).toInt()))
        {
            if(porttype==InputPort)
            {
                emit signalAddEdge(decode.at(1),decode.at(2).toUInt(),nodefullname,portid);
            }
            else
            {
                emit signalAddEdge(nodefullname,portid,decode.at(1),decode.at(2).toUInt());
            }
        }
    }
    this->setStyleSheet("QLabel { background-color : white; color : black; border: 2px solid yellow}");
}


XPortHead::XPortHead(QWidget *parent)
    : QLabel(parent)
{

}

XPortHead::XPortHead(QString text, QWidget *parent)
    : QLabel(text,parent)
{

}

void XPortHead::mousePressEvent(QMouseEvent *event)
{
    if(event->button()==Qt::RightButton)
    {
        Q_UNUSED(event);
        QMenu menu;
        menu.addAction("Change Port Num");
        QAction * selecteditem=menu.exec(QCursor::pos());
        if(selecteditem)
        {
            if(selecteditem->text()==QString("Change Port Num"))
            {
                portnum=QInputDialog::getInt(NULL,QString("Change %1 Port Num").arg(this->text()),QString("%1 Port Num  of %2").arg(this->text()).arg(nodefullname),portnum,0);
                emit signalResetPortNum(this->text(), portnum);
            }
        }
    }
}
