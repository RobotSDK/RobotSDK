#include "DPMAnnotatorWidget.h"

using namespace RobotSDK_Module;

DPMAnnotatorCheckBox::DPMAnnotatorCheckBox(QString attribute, int id, QWidget * parent)
    : QWidget(parent)
{
    QVBoxLayout * tmplayout=new QVBoxLayout;
    QStringList attributelist=attribute.split(":",QString::SkipEmptyParts);
    if(attributelist.size()!=2)
    {
        tmplayout->addWidget(new QLabel(QString("Invalid Attribute: %1").arg(attribute)));
    }
    else
    {
        tmplayout->addWidget(new QLabel(attributelist.at(0)));
    }
    QGroupBox * groupbox=new QGroupBox;
    tmplayout->addWidget(groupbox);
    layout=new QHBoxLayout;
    QStringList valuelist=attributelist.at(1).split(",",QString::SkipEmptyParts);
    int i,n=valuelist.size();
    for(i=0;i<n;i++)
    {
        QRadioButton * button=new QRadioButton(valuelist.at(i));
        layout->addWidget(button);
    }
    layout->addStretch();
    groupbox->setLayout(layout);
    this->setLayout(tmplayout);

    if(id<0||id>=n)
    {
        id=0;
    }
    QRadioButton * button=(QRadioButton *)(layout->itemAt(id)->widget());
    button->setChecked(1);
}

int DPMAnnotatorCheckBox::getSelection()
{
    int i,n=layout->count();
    for(i=0;i<n;i++)
    {
        QRadioButton * button=(QRadioButton *)(layout->itemAt(i)->widget());
        if(button->isChecked())
        {
            return i;
        }
    }
    return 0;
}

DPMAnnotatorWidget::DPMAnnotatorWidget(QWidget *parent) : QWidget(parent)
{
    QHBoxLayout * tmplayout=new QHBoxLayout;
    QSplitter * layout=new QSplitter(Qt::Horizontal);
    tmplayout->addWidget(layout);

    imageviewer=new QLabel("Image");
    imageviewer->setAlignment(Qt::AlignCenter);
    QScrollArea * scrollarea;
    scrollarea=new QScrollArea;
    scrollarea->setWidget(imageviewer);
    layout->addWidget(scrollarea);
    layout->setStretchFactor(0,2);

    QSplitter * vlayout=new QSplitter(Qt::Vertical);
    layout->addWidget(vlayout);
    layout->setStretchFactor(1,1);

    sampleinfo=new QLabel("Sample Info");
    vlayout->addWidget(sampleinfo);
    vlayout->setStretchFactor(0,0);

    sampleviewer=new QLabel("Sample");
    sampleviewer->setAlignment(Qt::AlignCenter);
    vlayout->addWidget(sampleviewer);
    vlayout->setStretchFactor(1,1);

    attributeslayout=new QTableWidget;
//    attributeslayout->horizontalHeader()->setStretchLastSection(1);
    vlayout->addWidget(attributeslayout);
    vlayout->setStretchFactor(2,2);

    QHBoxLayout * hlayout=new QHBoxLayout;
    hlayout->addStretch();
    QPushButton * next=new QPushButton("Next");
    hlayout->addWidget(next);
    connect(next,SIGNAL(clicked()),this,SLOT(slotNext()));
    QPushButton * skip=new QPushButton("Skip");
    hlayout->addWidget(skip);
    connect(skip,SIGNAL(clicked()),this,SLOT(slotSkip()));
    QWidget * widget=new QWidget;
    widget->setLayout(hlayout);
    vlayout->addWidget(widget);
    vlayout->setStretchFactor(3,0);

    this->setLayout(tmplayout);
}

DPMAnnotatorWidget::~DPMAnnotatorWidget()
{
    clearAttributes();
}

void DPMAnnotatorWidget::setAttributes(QString category, QString attribute)
{
    attributesmap.insert(category, attribute);
}

void DPMAnnotatorWidget::clearAttributes()
{
    attributeslayout->clear();
    attributeslayout->setRowCount(0);
    attributesmap.clear();
    imageviewer->setText("Image");
    sampleviewer->setText("Sampel");
    sampleinfo->setText("Sample Info");
}

void DPMAnnotatorWidget::showSample(cv::Mat image, int frameid, QString category, int id, cv::Rect rect, QString attributes)
{
    cv::Mat sample=image(rect);
    sample=sample.clone();
    QImage sampleimg(sample.data,sample.cols,sample.rows,sample.step,QImage::Format_RGB888);
    sampleviewer->setPixmap(QPixmap::fromImage(sampleimg));
    sampleviewer->resize(sampleimg.size());

    cv::rectangle(image,rect,cv::Scalar(255,0,0),3);
    QImage img(image.data,image.cols,image.rows,image.step,QImage::Format_RGB888);
    imageviewer->setPixmap(QPixmap::fromImage(img));
    imageviewer->resize(img.size());

    QString info=QString("Frame: %1, Category: %2, ID: %3, Geometry: [ %4, %5, %6, %7 ]").arg(frameid).arg(category).arg(id)
            .arg(rect.x).arg(rect.y).arg(rect.width).arg(rect.height);
    sampleinfo->setText(info);

    attributeslayout->clear();

    QList<QString> checkboxes=attributesmap.values(category);
    QStringList values=attributes.split(",",QString::SkipEmptyParts);
    int i,n=checkboxes.size();
    int j,m=values.size();
    attributeslayout->setColumnCount(2);
    attributeslayout->setRowCount(n);
    for(i=n-1,j=0;i>=0;i--,j++)
    {
        if(i<m)
        {
            attributeslayout->setCellWidget(j,0,new DPMAnnotatorCheckBox(checkboxes[i],values[i].toInt()));
        }
        else
        {
            attributeslayout->setCellWidget(j,0,new DPMAnnotatorCheckBox(checkboxes[i]));
        }
    }
    attributeslayout->resizeRowsToContents();
    attributeslayout->resizeColumnsToContents();
}

QString DPMAnnotatorWidget::getAttributes()
{
    QString result;
    if(annotateflag)
    {
        int i,n=attributeslayout->rowCount();
        for(i=0;i<n;i++)
        {
            DPMAnnotatorCheckBox * checkbox=(DPMAnnotatorCheckBox *)(attributeslayout->cellWidget(i,0));
            result+=QString("%1,").arg(checkbox->getSelection());
        }
        result.chop(1);
    }
    return result;
}

void DPMAnnotatorWidget::slotNext()
{
    annotateflag=1;
    emit signalNext();
}

void DPMAnnotatorWidget::slotSkip()
{
    annotateflag=0;
    emit signalNext();
}
