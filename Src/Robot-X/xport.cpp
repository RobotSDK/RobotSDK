#include "xport.h"

XPort::XPort(QWidget * parent)
    :QLabel(parent)
{
    setCursor(Qt::OpenHandCursor);
    this->setStyleSheet("QLabel { background-color : white; color : black; }");
    setAcceptDrops(1);
}

XPort::~XPort()
{

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
           info=QString("%1\nInput Port #%2").arg(nodefullname).arg(portid);
            break;
        case OutputPort:
            info=QString("%1\nOutput Port #%2").arg(nodefullname).arg(portid);
            break;
        default:
            break;
        }        
        QFontMetrics fm(QApplication::font());
        QPixmap pixmap(QSize(fm.width(info),fm.height()));
        drag->setHotSpot(QPoint(pixmap.width()/2,pixmap.height()/2));
        pixmap.fill(Qt::white);
        QPainter painter(&pixmap);
        painter.setPen(Qt::black);
        painter.drawText(pixmap.rect(), Qt::AlignCenter, info);
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
                    this->setStyleSheet("QLabel { background-color : red; color : black; }");
                }
            }
            else
            {
                this->setStyleSheet("QLabel { background-color : green; color : black; }");
            }
            event->acceptProposedAction();
        }
    }
}

void XPort::dragLeaveEvent(QDragLeaveEvent *event)
{
    Q_UNUSED(event);
    this->setStyleSheet("QLabel { background-color : white; color : black; }");
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
                    this->setStyleSheet("QLabel { background-color : red; color : black; }");
                }
            }
            else
            {
                this->setStyleSheet("QLabel { background-color : green; color : black; }");
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
    this->setStyleSheet("QLabel { background-color : white; color : black; }");
}
