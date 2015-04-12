#include "xport.h"

XPort::XPort(QWidget * parent)
    :QLabel(parent)
{
    setCursor(Qt::OpenHandCursor);
    QPalette palette=this->palette();
    palette.setColor(this->backgroundRole(), Qt::white);
    this->setPalette(palette);
}

XPort::~XPort()
{

}

void XPort::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton
            && this->geometry().contains(event->pos()))
    {
        setCursor(Qt::ClosedHandCursor);
        QDrag *drag = new QDrag(this);
        QMimeData *mimeData = new QMimeData;
        mimeData->setText(QString("%1~~%2~~%3").arg(porttype).arg(nodefullname).arg(portid));
        drag->setMimeData(mimeData);
        QLabel * label;
        switch (porttype)
        {
        case InputPort:
            label=new QLabel(QString("%1\nInput Port #%2").arg(nodefullname).arg(portid));
            break;
        case OutputPort:
            label=new QLabel(QString("%1\nOutput Port #%2").arg(nodefullname).arg(portid));
            break;
        default:
            break;
        }
        label->setAlignment(Qt::AlignCenter);
        QPixmap pixmap(label->size());
        pixmap.fill(Qt::white);
        QPainter painter(&pixmap);
        painter.setPen(Qt::black);
        painter.drawText(label->rect(), Qt::AlignCenter, label->text());
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
        if(decode.size()==3&&porttype!=PORTTYPE(decode.at(0).toInt()))
        {
            QPalette palette=this->palette();
            palette.setColor(this->backgroundRole(), Qt::green);
            this->setPalette(palette);
            event->setAccepted(1);
        }
        else
        {
            QPalette palette=this->palette();
            palette.setColor(this->backgroundRole(), Qt::red);
            this->setPalette(palette);
            event->setAccepted(0);
        }
    }
}

void XPort::dragLeaveEvent(QDragLeaveEvent *event)
{
    Q_UNUSED(event);
    QPalette palette=this->palette();
    palette.setColor(this->backgroundRole(), Qt::white);
    this->setPalette(palette);
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
    QPalette palette=this->palette();
    palette.setColor(this->backgroundRole(), Qt::white);
    this->setPalette(palette);
}
