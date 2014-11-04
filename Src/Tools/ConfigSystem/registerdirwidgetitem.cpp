#include "registerdirwidgetitem.h"

#ifdef Q_OS_LINUX

QString getRegValue(QString regValue)
{

    QFile file;
    if(regValue==PATHATTR)
    {
        file.setFileName(ROBOTSDKPATHSH);
    }
    else
    {
        file.setFileName(ROBOTSDKSH);
    }
    if(!file.exists())
    {
        file.open(QIODevice::WriteOnly|QIODevice::Text);
        file.close();
    }
    if(file.open(QIODevice::ReadOnly|QIODevice::Text))
    {
        while(!file.atEnd())
        {
            QString tmpline=file.readLine();
            if(tmpline.startsWith("export"))
            {
                QString tmpstr=tmpline.mid(6).trimmed();
                if(tmpstr.startsWith(QString("%1=").arg(regValue)))
                {
                    file.close();
                    int equalindex=tmpstr.indexOf("=");
                    return tmpstr.mid(equalindex+1).trimmed();
                }
            }
        }
        file.close();
        if(regValue==PATHATTR)
        {
            setRegValue(regValue,QString());
            return getRegValue(regValue);
        }
        else
        {
            return QString();
        }
    }
    else
    {
        return QString();
    }
}

bool setRegValue(QString regValue, QString data)
{
    if(regValue!=PATHATTR&&data.isEmpty())
    {
        return 0;
    }
    QFile file;
    if(regValue==PATHATTR)
    {        
        file.setFileName(ROBOTSDKPATHSH);
    }
    else
    {
        file.setFileName(ROBOTSDKSH);
    }
    if(!file.exists())
    {
        file.open(QIODevice::WriteOnly|QIODevice::Text);
        file.close();
    }
    QStringList tmplist;
    if(file.open(QIODevice::ReadOnly|QIODevice::Text))
    {        
        bool flag=1;
        while(!file.atEnd())
        {
            QString tmpline=file.readLine();
            if(tmpline.startsWith("export"))
            {
                QString tmpstr=tmpline.mid(6).trimmed();
                if(tmpstr.startsWith(QString("%1=").arg(regValue)))
                {
                    flag=0;
                    tmpline=QString("export %1=%2\n").arg(regValue).arg(data);
                }
            }
            tmplist.push_back(tmpline);
        }
        file.close();
        if(flag)
        {
            if(regValue==PATHATTR)
            {
                tmplist.push_back(QString("export %1=$%1:%2\n").arg(regValue).arg(data));
            }
            else
            {
                tmplist.push_back(QString("export %1=%2\n").arg(regValue).arg(data));
            }
        }
    }
    if(file.open(QIODevice::WriteOnly|QIODevice::Text))
    {
        int i,n=tmplist.size();
        for(i=0;i<n;i++)
        {
            file.write(tmplist[i].toUtf8());
        }
        file.close();
        setenv(regValue.toUtf8().constData(),data.toUtf8().constData(),1);
        return 1;
    }
    else
    {
        return 0;
    }
}

bool deleteRegValue(QString regValue)
{
    QFile file(ROBOTSDKSH);
    if(!file.exists())
    {
        file.open(QIODevice::WriteOnly|QIODevice::Text);
        file.close();
    }
    QStringList tmplist;
    if(file.open(QIODevice::ReadOnly|QIODevice::Text))
    {        
        while(!file.atEnd())
        {
            QString tmpline=file.readLine();
            bool flag=1;
            if(tmpline.startsWith("export"))
            {
                QString tmpstr=tmpline.mid(6).trimmed();
                if(tmpstr.startsWith(QString("%1=").arg(regValue)))
                {
                    flag=0;
                }
            }
            if(flag)
            {
                tmplist.push_back(tmpline);
            }
        }
        file.close();
    }
    if(file.open(QIODevice::WriteOnly|QIODevice::Text))
    {
        int i,n=tmplist.size();
        for(i=0;i<n;i++)
        {
            file.write(tmplist[i].toUtf8());
        }
        file.close();
        unsetenv(regValue.toUtf8().constData());
        return 1;
    }
    else
    {
        return 0;
    }
}

#elif defined(Q_OS_WIN)

#pragma comment(lib, "Advapi32.lib")

QString getRegValue(HKEY hkey, QString subkey, QString regValue)
{
	static char databuffer[REGDATABUFFERSIZE];
	DWORD datasize=REGDATABUFFERSIZE;
	long result=RegGetValueA(hkey,subkey.toUtf8().data(),regValue.toUtf8().data(),RRF_RT_ANY,NULL,databuffer,&datasize);
	if(result==ERROR_SUCCESS)
	{
		return QString(databuffer);
	}
	else
	{
		return QString();
	}
}

bool setRegValue(HKEY hkey, QString subkey, QString regValue, DWORD dataType, QString data)
{
	DWORD datasize=data.size();
	long result=RegSetKeyValueA(hkey,subkey.toUtf8().data(),regValue.toUtf8().data(),dataType,data.toUtf8().data(),	datasize);
	if(result==ERROR_SUCCESS)
	{
		_putenv_s(regValue.toUtf8().constData(),data.toUtf8().constData());
		return 1;
	}
	else
	{
		return 0;
	}
}

bool deleteRegValue(HKEY hkey, QString subkey, QString regValue)
{
	long result=RegDeleteKeyValueA(hkey,subkey.toUtf8().data(),regValue.toUtf8().data());
	if(result==ERROR_SUCCESS)
	{
		_putenv_s(regValue.toUtf8().constData(),"");
		return 1;
	}
	else
	{
		return 0;
	}
}

#endif

RegisterDirWidgetItem::RegisterDirWidgetItem(QWidget * parent, QString qstrRegisterValue, bool editable)
	: QWidget(parent)
{
	ui.setupUi(this);
#ifdef Q_OS_LINUX
    ui.Prefix->setText("$");
    ui.Suffix->clear();
#elif defined(Q_OS_WIN)
    ui.Prefix->setText("%");
    ui.Suffix->setText("%");
#endif
    ui.RegisterValue->setText(qstrRegisterValue);
    doc=NULL;

	ui.RegisterTable->sortByColumn(0,Qt::AscendingOrder);

	connect(ui.Add,SIGNAL(clicked()),this,SLOT(addRegisterDirTagSlot()));
	connect(ui.Tag,SIGNAL(returnPressed()),this,SLOT(addRegisterDirTagSlot()));
	connect(ui.Delete,SIGNAL(clicked()),this,SLOT(deleteRegisterDirTagSlot()));
	connect(ui.Clear,SIGNAL(clicked()),this,SLOT(clearRegisterDirTagSlot()));

    ui.Add->setEnabled(editable);
    ui.Tag->setEnabled(editable);
    ui.Delete->setEnabled(editable);
    ui.Clear->setEnabled(editable);
    ui.Path->setEnabled(editable);

    if(editable)
    {
        connect(ui.RegisterTable,SIGNAL(cellDoubleClicked(int,int)),this,SLOT(processDoubleClickedSlot(int,int)));
    }
}

RegisterDirWidgetItem::~RegisterDirWidgetItem()
{

}

void RegisterDirWidgetItem::setQDomDocument(QDomDocument * pDoc)
{
	doc=pDoc;
}

void RegisterDirWidgetItem::loadRegisterDir(QDomElement & rootElem)
{
	QString registervalue=ui.RegisterValue->text();
	QDomElement curelem=rootElem.firstChildElement(registervalue);
	if(curelem.isNull())
	{
		curelem=rootElem.appendChild(doc->createElement(registervalue)).toElement();
	}
	ui.Path->setChecked(bool(curelem.attribute(PATHATTR,QString("0")).toUInt()));
	QDomNodeList nodelist=curelem.childNodes();
	int i,n=nodelist.size();
	ui.RegisterTable->setRowCount(n);
    ui.RegisterTable->setSortingEnabled(0);
	for(i=0;i<n;i++)
	{
		if(nodelist.at(i).isElement())
		{
			QString tag=nodelist.at(i).toElement().nodeName();
			QString dir=nodelist.at(i).toElement().text();
			ui.RegisterTable->setItem(i,0,new QTableWidgetItem(tag));
			ui.RegisterTable->setItem(i,1,new QTableWidgetItem(dir));
            if(!dir.isEmpty()&&QDir(dir).exists())
			{
				ui.RegisterTable->item(i,1)->setBackgroundColor(QColor::fromRgbF(1,1,1,0.5));
			}
			else
			{
				ui.RegisterTable->item(i,1)->setBackgroundColor(QColor::fromRgbF(1,0,0,0.5));
			}
		}
	}
    ui.RegisterTable->setSortingEnabled(1);
    ui.RegisterTable->sortByColumn(0,Qt::AscendingOrder);
}

void RegisterDirWidgetItem::saveRegisterDir(QDomElement & rootElem)
{
	QString registervalue=ui.RegisterValue->text();
	QDomElement curelem=rootElem.firstChildElement(registervalue);
	if(curelem.isNull())
	{
		curelem=rootElem.appendChild(doc->createElement(registervalue)).toElement();
	}
    curelem.setAttribute(PATHATTR, uint (ui.Path->isChecked()));
	int i,n=ui.RegisterTable->rowCount();
	for(i=0;i<n;i++)
	{
		QString tag=ui.RegisterTable->item(i,0)->text();
		QString dir=ui.RegisterTable->item(i,1)->text();
		QDomElement tmpelem=curelem.firstChildElement(tag);
		if(tmpelem.isNull())
		{
			tmpelem=curelem.appendChild(doc->createElement(tag)).toElement();
			tmpelem.appendChild(doc->createTextNode(dir));
		}
		else
		{
			QDomNode tmpnode=tmpelem.firstChild();
			while(!tmpnode.isNull())
			{
				if(tmpnode.isText())
				{
					tmpnode.setNodeValue(dir);
					break;
				}
				tmpnode=tmpnode.nextSibling();
			}
		}
	}
}

void RegisterDirWidgetItem::registerDir()
{
	QString registervalue=ui.RegisterValue->text();
	QString registerdata;
	int i,n=ui.RegisterTable->rowCount();
	if(n>0)
	{
        bool flag=registerPath(registervalue,ui.Path->isChecked());
        registerdata.clear();
        for(i=0;i<n;i++)
        {
            QString tmpdir=ui.RegisterTable->item(i,1)->text();
            if(!tmpdir.isEmpty()&&QDir(tmpdir).exists())
            {
            #ifdef Q_OS_LINUX
                registerdata=registerdata+QString("%1:").arg(tmpdir);
            #elif defined(Q_OS_WIN)
                registerdata=registerdata+QString("%1;").arg(tmpdir);
            #endif
            }
        }
    #ifdef Q_OS_LINUX
        if(registerdata.endsWith(":"))
    #elif defined(Q_OS_WIN)
        if(registerdata.endsWith(";"))
    #endif
        {
            registerdata.truncate(registerdata.size()-1);
        }
    #ifdef Q_OS_LINUX
        if(setRegValue(registervalue,registerdata))
    #elif defined(Q_OS_WIN)
        if(setRegValue(HKEY_LOCAL_MACHINE,REGENVPATH,registervalue,REG_SZ,registerdata))
    #endif
		{
			QMessageBox::information(this,QString("RegisterDir"),QString("Register %1 Successfully!").arg(registervalue));
			if(!flag)
			{
				registerPath(registervalue,ui.Path->isChecked());
			}
		}
		else
		{
			QMessageBox::information(this,QString("RegisterDir"),QString("Register %1 Unsuccessfully!").arg(registervalue));
		}		
	}
	else
	{
		registerPath(registervalue,0);
    #ifdef Q_OS_LINUX
        if(deleteRegValue(registervalue))
    #elif defined(Q_OS_WIN)
        if(deleteRegValue(HKEY_LOCAL_MACHINE,REGENVPATH,registervalue))
    #endif
		{
			QMessageBox::information(this,QString("RegisterDir"),QString("Annual %1 Successfully!").arg(registervalue));
		}
		else
		{
			QMessageBox::information(this,QString("RegisterDir"),QString("Annual %1 Unsuccessfully!").arg(registervalue));
		}
	}
}

bool RegisterDirWidgetItem::registerPath(QString registerValue,bool addFlag)
{
#ifdef Q_OS_LINUX
    QString regkey=QString("$%1:").arg(registerValue);
    QString regkeyex=QString();
    QString pathtmp=getRegValue(PATHATTR);
#elif defined(Q_OS_WIN)
	QString regkey=QString("%%1%;").arg(registerValue);
    QString regkeyex=QString("%1;").arg(getenv(registerValue.toUtf8().data()));
    if(regkeyex.size()<=1)
    {
        return 0;
    }
    QString pathtmp=getRegValue(HKEY_LOCAL_MACHINE,REGENVPATH,PATHATTR);
#endif
    int regkeyindex;
	int regkeyexindex;
	regkeyindex=pathtmp.indexOf(regkey);
	QString tmpkey;
	if(regkeyindex<0)
    {
    #ifdef Q_OS_LINUX
        regkeyexindex=-1;
        tmpkey=QString();
    #elif defined(Q_OS_WIN)
		regkeyexindex=pathtmp.indexOf(regkeyex);
        tmpkey=regkeyex;
    #endif		
	}
	else
	{
		tmpkey=regkey;
	}
	if(addFlag)
	{	
		if(regkeyindex<0&&regkeyexindex<0)
		{
        #ifdef Q_OS_LINUX
            if(pathtmp.endsWith(":"))
        #elif defined(Q_OS_WIN)
			if(pathtmp.endsWith(";"))
        #endif
			{
				pathtmp=QString("%1%2").arg(pathtmp).arg(regkey);
			}
			else
			{
            #ifdef Q_OS_LINUX
                pathtmp=QString("%1:%2").arg(pathtmp).arg(regkey);
            #elif defined(Q_OS_WIN)
                pathtmp=QString("%1;%2").arg(pathtmp).arg(regkey);
            #endif
			}
        #ifdef Q_OS_LINUX
            if(setRegValue(PATHATTR,pathtmp))
        #elif defined(Q_OS_WIN)
            if(setRegValue(HKEY_LOCAL_MACHINE,REGENVPATH,PATHATTR,REG_EXPAND_SZ,pathtmp))
        #endif
			{
				QMessageBox::information(this,QString("RegisterPath"),QString("Add %%1% to %PATH% Successfully.").arg(registerValue));
			}
			else
			{
				QMessageBox::information(this,QString("RegisterPath"),QString("Add %%1% to %PATH% Unsuccessfully.").arg(registerValue));
			}
		}
		else if(regkeyindex>=0)
		{
			QMessageBox::information(this,QString("RegisterPath"),QString("%%1% is already in %PATH%.").arg(registerValue));
		}
		else if(regkeyexindex>=0)
		{
			pathtmp.remove(tmpkey);	
			QMessageBox::information(this,QString("RegisterPath"),QString("%%1% is already in %PATH%, but is expanded.").arg(registerValue));
        #ifdef Q_OS_LINUX
            if(pathtmp.endsWith(":"))
        #elif defined(Q_OS_WIN)
            if(pathtmp.endsWith(";"))
        #endif
            {
                pathtmp=QString("%1%2").arg(pathtmp).arg(regkey);
            }
            else
            {
            #ifdef Q_OS_LINUX
                pathtmp=QString("%1:%2").arg(pathtmp).arg(regkey);
            #elif defined(Q_OS_WIN)
                pathtmp=QString("%1;%2").arg(pathtmp).arg(regkey);
            #endif
            }
        #ifdef Q_OS_LINUX
            if(setRegValue(PATHATTR,pathtmp))
        #elif defined(Q_OS_WIN)
            if(setRegValue(HKEY_LOCAL_MACHINE,REGENVPATH,PATHATTR,REG_EXPAND_SZ,pathtmp))
        #endif
			{
				QMessageBox::information(this,QString("RegisterPath"),QString("Add %%1% to %PATH% Successfully.").arg(registerValue));
			}
			else
			{
				QMessageBox::information(this,QString("RegisterPath"),QString("Add %%1% to %PATH% Unsuccessfully.").arg(registerValue));
			}			
		}
	}
	else
	{
		if(regkeyindex>=0||regkeyexindex>=0)
		{
			pathtmp.remove(tmpkey);		
        #ifdef Q_OS_LINUX
            if(setRegValue(PATHATTR,pathtmp))
        #elif defined(Q_OS_WIN)
            if(setRegValue(HKEY_LOCAL_MACHINE,REGENVPATH,PATHATTR,REG_EXPAND_SZ,pathtmp))
        #endif
			{
				QMessageBox::information(this,QString("RegisterPath"),QString("Remove %%1% from %PATH% Successfully!").arg(registerValue));
			}
			else
			{
				QMessageBox::information(this,QString("RegisterPath"),QString("Remove %%1% from %PATH% Successfully!").arg(registerValue));
			}
		}
	}
	return 1;
}

void RegisterDirWidgetItem::addRegisterDirTagSlot()
{
	QString dirtag=ui.Tag->text();
	ui.Tag->clear();
	if(dirtag.size()>0)
	{
		int i,n=ui.RegisterTable->rowCount();
		for(i=0;i<n;i++)
		{
			if(ui.RegisterTable->item(i,0)->text()==dirtag)
			{
				break;
			}
		}
		if(i==n)
		{
			ui.RegisterTable->setSortingEnabled(0);
			ui.RegisterTable->insertRow(n);
			ui.RegisterTable->setItem(n,0,new QTableWidgetItem(dirtag));
			ui.RegisterTable->setItem(n,1,new QTableWidgetItem(QString()));
			ui.RegisterTable->setCurrentCell(n,0);
			ui.RegisterTable->setSortingEnabled(1);
			ui.RegisterTable->sortByColumn(0,Qt::AscendingOrder);
		}
		else
		{
			ui.RegisterTable->setCurrentCell(i,0);
		}
	}
}

void RegisterDirWidgetItem::deleteRegisterDirTagSlot()
{
	int currow=ui.RegisterTable->currentRow();
	if(currow>=0)
	{
		ui.RegisterTable->removeRow(currow);
	}
}

void RegisterDirWidgetItem::clearRegisterDirTagSlot()
{
	int i,n=ui.RegisterTable->rowCount();
	for(i=0;i<n;i++)
	{
		ui.RegisterTable->removeRow(i);
	}
}

void RegisterDirWidgetItem::processDoubleClickedSlot(int row, int column)
{
	QString inputvalue=ui.RegisterTable->item(row,column)->text();
    QDir dir;
	switch(column)
	{
	case 0:
		inputvalue=QInputDialog::getText(this,QString("Input Value"),QString("Tag"),QLineEdit::Normal,inputvalue);
		if(inputvalue.size()>0)
		{
			int i,n=ui.RegisterTable->rowCount();
			bool flag=1;
			for(i=0;i<n;i++)
			{
				if(ui.RegisterTable->item(i,0)->text()==inputvalue)
				{
					flag=0;
					break;
				}
			}
			if(flag)
			{
				ui.RegisterTable->item(row,column)->setText(inputvalue);
				ui.RegisterTable->sortByColumn(0,Qt::AscendingOrder);
			}
		}
		break;
	case 1:
        dir.setPath(inputvalue);
        if(dir.exists())
        {
            inputvalue=QFileDialog::getExistingDirectory(this,QString("Select Directory"),inputvalue);
        }
        else
        {
            inputvalue=QFileDialog::getExistingDirectory(this,QString("Select Directory"));
        }
		if(inputvalue.size()>0)
		{
			ui.RegisterTable->item(row,column)->setText(inputvalue);
			if(QDir(inputvalue).exists())
			{
				ui.RegisterTable->item(row,1)->setBackgroundColor(QColor::fromRgbF(1,1,1,0.5));
			}
			else
			{
				ui.RegisterTable->item(row,1)->setBackgroundColor(QColor::fromRgbF(1,0,0,0.5));
			}
		}
		break;
	}
}

QString RegisterDirWidgetItem::getRegisterValue()
{
	return ui.RegisterValue->text();
}

bool RegisterDirWidgetItem::hasData()
{
	return ui.RegisterTable->rowCount()>0;
}
