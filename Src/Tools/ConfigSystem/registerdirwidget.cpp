#include "registerdirwidget.h"

RegisterDirWidget::RegisterDirWidget(QWidget * parent, QDomDocument * qDoc, QString qstrRegisterName)
	: QWidget(parent)
{
	ui.setupUi(this);
	ui.RegisterName->setText(qstrRegisterName);
	doc=qDoc;

	connect(ui.Register,SIGNAL(clicked()),this,SLOT(registerDir()));
}

RegisterDirWidget::RegisterDirWidget(QWidget * parent, QDomDocument * qDoc, QString qstrRegisterName, QStringList qstrlsRegisterValue, bool editable)
	: QWidget(parent)
{
	ui.setupUi(this);
	ui.RegisterName->setText(qstrRegisterName);
	doc=qDoc;

	connect(ui.Register,SIGNAL(clicked()),this,SLOT(registerDir()));

	int i,n=qstrlsRegisterValue.size();
	for(i=0;i<n;i++)
	{
        this->addItem(qstrlsRegisterValue.at(i),editable);
	}
}

RegisterDirWidget::~RegisterDirWidget()
{

}

void RegisterDirWidget::setQDomDocument(QDomDocument * pDoc)
{
	doc=pDoc;
	int i,n=ui.RegisterTab->count();
	for(i=0;i<n;i++)
	{
		RegisterDirWidgetItem * item=(RegisterDirWidgetItem *)(ui.RegisterTab->widget(i));
		item->setQDomDocument(doc);
	}
}

void RegisterDirWidget::addItem(QString itemName, bool editable)
{
	QString registervalue=QString("%1_%2").arg(ui.RegisterName->text(),itemName);
    RegisterDirWidgetItem * registeritem=new RegisterDirWidgetItem(this,registervalue,editable);
	ui.RegisterTab->addTab(registeritem,itemName);
	registeritem->setQDomDocument(doc);
}

void RegisterDirWidget::loadRegisterDir(QDomElement & rootElem)
{
	QString registername=ui.RegisterName->text();
	QDomElement curelem=rootElem.firstChildElement(registername);
	if(curelem.isNull())
	{
		curelem=rootElem.appendChild(doc->createElement(registername)).toElement();
	}
	int i,n=ui.RegisterTab->count();
	for(i=0;i<n;i++)
	{
		RegisterDirWidgetItem * item=(RegisterDirWidgetItem *)(ui.RegisterTab->widget(i));
		item->loadRegisterDir(curelem);
	}
}

void RegisterDirWidget::saveRegisterDir(QDomElement & rootElem)
{
	QString registername=ui.RegisterName->text();
	QDomElement curelem=rootElem.firstChildElement(registername);
	if(curelem.isNull())
	{
		curelem=rootElem.appendChild(doc->createElement(registername)).toElement();
	}
	int i,n=ui.RegisterTab->count();
	for(i=0;i<n;i++)
	{
		RegisterDirWidgetItem * item=(RegisterDirWidgetItem *)(ui.RegisterTab->widget(i));
		item->saveRegisterDir(curelem);
	}
}

void RegisterDirWidget::registerDir()
{
	int i,n=ui.RegisterTab->count();
	for(i=0;i<n;i++)
	{
		RegisterDirWidgetItem * item=(RegisterDirWidgetItem *)(ui.RegisterTab->widget(i));
		item->registerDir();
	}
}

QStringList RegisterDirWidget::getRegisterValues()
{
	int i,n=ui.RegisterTab->count();
	QStringList result;
	for(i=0;i<n;i++)
	{
		result<<((RegisterDirWidgetItem *)(ui.RegisterTab->widget(i)))->getRegisterValue();
	}
	return result;
}

bool RegisterDirWidget::hasData(int id)
{
	RegisterDirWidgetItem *tmpitem=(RegisterDirWidgetItem *)(ui.RegisterTab->widget(id));
	if(tmpitem!=NULL)
	{
		return tmpitem->hasData();
	}
	else
	{
		return 0;
	}
}
