#include "scaninterfacefunction.h"

bool InterfaceFunctionNode::scan(QTextStream & textstream, QString & classline)
{
	classline.clear();
	comment.clear();
	parametertypes.clear();
	parameternames.clear();
	while(!textstream.atEnd())
	{
		QString tmpline=textstream.readLine().trimmed();
		if(tmpline.startsWith("class "))
		{
			classline=tmpline;
			return 0;
		}
		if(tmpline.startsWith("/*!")&&tmpline.contains("\\typedef "))
		{
			while(!textstream.atEnd())
			{
				tmpline=textstream.readLine().trimmed();
				if(tmpline.startsWith("*/"))
				{
					break;
				}
				comment.push_back(tmpline);
			}
		}
		else if(tmpline.startsWith("typedef ")&&tmpline.contains("(*")&&tmpline.contains(")(")&&tmpline.contains(");"))
		{
			int startid=tmpline.indexOf(" ");
			int endid=tmpline.indexOf("(*");
			returnvalue=tmpline.mid(startid,endid-startid).trimmed();
			startid=endid+2;
			endid=tmpline.indexOf(")(");
			functionname=tmpline.mid(startid,endid-startid).trimmed();
			if(functionname.endsWith("Fptr"))
			{
				functionname=functionname.left(functionname.size()-4);
			}
			startid=endid+2;
			endid=tmpline.indexOf(");");
			tmpline=tmpline.mid(startid,endid-startid).trimmed();
			QList<QString> parameters=tmpline.split(",",QString::SkipEmptyParts);
			int i,n=parameters.size();
			for(i=0;i<n;i++)
			{
				tmpline=parameters.at(i).trimmed();
				int sepid=tmpline.lastIndexOf(" ");
				parametertypes.push_back(tmpline.left(sepid).trimmed());
				parameternames.push_back(tmpline.mid(sepid).trimmed());
			}
			return 1;
		}
	}
	return 0;
}

void InterfaceFunctionNode::store(QDomDocument * doc, QDomElement & classRoot)
{
	QDomElement interfacenode=classRoot.appendChild(doc->createElement(functionname)).toElement();
	QDomElement curelem;
	curelem=interfacenode.appendChild(doc->createElement("Comment")).toElement();
	{
		int i,n=comment.size();
		for(i=0;i<n;i++)
		{
			curelem.appendChild(doc->createElement("Text")).appendChild(doc->createTextNode(comment[i]));
		}
	}
	curelem=interfacenode.appendChild(doc->createElement("Return")).toElement();
	curelem.appendChild(doc->createTextNode(returnvalue));
	curelem=interfacenode.appendChild(doc->createElement("Parameters")).toElement();
	{
		int i,n=parameternames.size();
		for(i=0;i<n;i++)
		{
			curelem.appendChild(doc->createElement(parameternames[i])).appendChild(doc->createTextNode(parametertypes[i]));
		}
	}
}

void InterfaceFunctionNode::load(QDomElement & interfaceRoot)
{
	comment.clear();
	parametertypes.clear();
	parameternames.clear();
	functionname=interfaceRoot.nodeName();
	QDomElement curelem;
	curelem=interfaceRoot.firstChildElement("Comment").firstChildElement();
	while(!curelem.isNull())
	{
		comment.push_back(curelem.text());
		curelem=curelem.nextSiblingElement();
	}
	curelem=interfaceRoot.firstChildElement("Return");
	returnvalue=curelem.text();
	curelem=interfaceRoot.firstChildElement("Parameters").firstChildElement();
	while(!curelem.isNull())
	{
		parameternames.push_back(curelem.nodeName());
		parametertypes.push_back(curelem.text());
		curelem=curelem.nextSiblingElement();
	}
}

bool ClassNode::scan(QTextStream & textstream,QString fileName)
{
	filename=fileName;
	interfacefunctions.clear();
	static QString classline=QString();
	QString tmpline;
	while(!textstream.atEnd())
	{
		if(classline.size()==0)
		{
			tmpline=textstream.readLine().trimmed();
		}
		else
		{
			tmpline=classline;
		}
		if(tmpline.startsWith("class "))
		{
			if(tmpline.count("public ")!=1||tmpline.count("private ")!=0||tmpline.count("protected ")!=0||tmpline.count(",")!=0)
			{
				continue;
			}
			int startid=tmpline.indexOf(" ");
			int endid=tmpline.indexOf(":");
			classname=tmpline.mid(startid,endid-startid).trimmed();
			startid=tmpline.lastIndexOf(" ");
			parentclass=tmpline.mid(startid).trimmed();
			InterfaceFunctionNode interfacefunction;
			while(interfacefunction.scan(textstream,classline))
			{
				interfacefunctions.push_back(interfacefunction);
			}
			return 1;
		}
	}
	classline=QString();
	return 0;
}

void ClassNode::store(QDomDocument * doc, QDomElement & classRoot)
{
	QDomElement classnode=classRoot.appendChild(doc->createElement(classname)).toElement();
	QDomElement curelem;
	curelem=classnode.appendChild(doc->createElement("File")).toElement();
	curelem.appendChild(doc->createTextNode(filename));
	curelem=classnode.appendChild(doc->createElement("Parent")).toElement();
	curelem.appendChild(doc->createTextNode(parentclass));
	curelem=classnode.appendChild(doc->createElement("Interface")).toElement();
	int i,n=interfacefunctions.size();
	for(i=0;i<n;i++)
	{
		interfacefunctions[i].store(doc,curelem);
	}
	return;
}

void ClassNode::load(QDomElement & classRoot)
{
	interfacefunctions.clear();
	classname=classRoot.nodeName();
	QDomElement curelem;
	curelem=classRoot.firstChildElement("File");
	filename=curelem.text();
	curelem=classRoot.firstChildElement("Parent");
	parentclass=curelem.text();
	curelem=classRoot.firstChildElement("Interface");
	
	InterfaceFunctionNode interfacefunction;
	QDomElement interfaceroot=curelem.firstChildElement();
	while(!interfaceroot.isNull())
	{
		interfacefunction.load(interfaceroot);
		interfacefunctions.push_back(interfacefunction);
		interfaceroot=interfaceroot.nextSiblingElement();
	}
}

ScanInterfaceFunction::ScanInterfaceFunction(QWidget *parent)
	: QWidget(parent)
{
	ui.setupUi(this);

    QString RobotSDK_Kernel=ROBOTSDKKERNEL;
	if(RobotSDK_Kernel.size()==0)
	{
		QMessageBox::information(this,"Error",QString("Environment variable %1 does not exist. Please run Configuration first.").arg(ROBOTSDKKERNEL));
		exit(0);
	}
	ui.KernelDir->setText(QString("%1/include").arg(RobotSDK_Kernel));

    QString RobotSDK_Moduledev=ROBOTSDKMODULEDEV;
    if(RobotSDK_Moduledev.size()==0)
	{
        QMessageBox::information(this,"Error",QString("Environment variable %1 does not exist. Please run Configuration first.").arg(ROBOTSDKMODULEDEV));
		exit(0);
	}
    ui.LibraryDir->setText(RobotSDK_Moduledev);

	bool flag=1;
	flag&=bool(connect(ui.Scan,SIGNAL(clicked()),this,SLOT(scanKernelDirSlot())));
	flag&=bool(connect(ui.ClassesFiles,SIGNAL(itemSelectionChanged()),this,SLOT(showInterfaceFunctionsSlot())));
	flag&=bool(connect(ui.InterfaceFunctions,SIGNAL(itemSelectionChanged()),this,SLOT(showDetailInformationSlot())));
	flag&=bool(connect(ui.BrowseParamsHeader,SIGNAL(clicked()),this,SLOT(browseParamsHeaderSlot())));
	flag&=bool(connect(ui.ClearParamsHeader,SIGNAL(clicked()),this,SLOT(clearParamsHeaderSlot())));
	flag&=bool(connect(ui.BrowseOutputDataHeader,SIGNAL(clicked()),this,SLOT(browseOutputDataHeaderSlot())));
	flag&=bool(connect(ui.ClearOutputDataHeader,SIGNAL(clicked()),this,SLOT(clearOutputDataHeaderSlot())));
	flag&=bool(connect(ui.BrowseVarsHeader,SIGNAL(clicked()),this,SLOT(browseVarsHeaderSlot())));
	flag&=bool(connect(ui.ClearVarsHeader,SIGNAL(clicked()),this,SLOT(clearVarsHeaderSlot())));
	flag&=bool(connect(ui.AddInputDataHeader,SIGNAL(clicked()),this,SLOT(addInputDataParamsHeadersSlot())));
	flag&=bool(connect(ui.BrowseInputDataHeader,SIGNAL(clicked()),this,SLOT(browseInputDataParamsHeadersSlot())));
	flag&=bool(connect(ui.DeleteInputDataHeader,SIGNAL(clicked()),this,SLOT(deleteInputDataParamsHeadersSlot())));
    flag&=bool(connect(ui.ClearInputDataHeader,SIGNAL(clicked()),this,SLOT(clearInputDataParamsHeadersSlot())));
	flag&=bool(connect(ui.AddInputPort,SIGNAL(clicked()),this,SLOT(addInputPortsSizeSlot())));
	flag&=bool(connect(ui.DeleteInputPort,SIGNAL(clicked()),this,SLOT(deleteInputPortsSizeSlot())));
    flag&=bool(connect(ui.ClearInputPort,SIGNAL(clicked()),this,SLOT(clearInputPortsSizeSlot())));
    flag&=bool(connect(ui.InputDataheadersList,SIGNAL(cellDoubleClicked(int, int)),this,SLOT(processDoubleClickInputParamsDataSlot(int,int))));
	flag&=bool(connect(ui.InputPortsSize,SIGNAL(cellDoubleClicked(int, int)),this,SLOT(processDoubleClickInputPortSizeSlot(int,int))));
	flag&=bool(connect(ui.AddExFunc,SIGNAL(clicked()),this,SLOT(addExFuncSlot())));
	flag&=bool(connect(ui.DeleteExFunc,SIGNAL(clicked()),this,SLOT(deleteExFuncSlot())));
	flag&=bool(connect(ui.ClearExFuncs,SIGNAL(clicked()),this,SLOT(clearExFuncsSlot())));
	flag&=bool(connect(ui.BrowseOutputDir,SIGNAL(clicked()),this,SLOT(browseOutputDirSlot())));
	flag&=bool(connect(ui.Create,SIGNAL(clicked()),this,SLOT(createLibraryFilesSlot())));
	
    ui.OutputLabel->setText("Choose Output Project");

	doc=new QDomDocument(INTERFACEFUNCTIONS);
    QString filename=QString("%1/%2.xml").arg(ui.KernelDir->text()).arg(INTERFACEFUNCTIONS);
	QFile file(filename);
	if(!file.open(QIODevice::ReadOnly|QIODevice::Text))
	{
		doc->appendChild(doc->createElement(INTERFACEFUNCTIONS));
	}
	else
	{
		if(!doc->setContent(&file))
		{
			delete doc;
			file.close();
			QMessageBox::information(this,QString("Error"),QString("%1.xml broken").arg(INTERFACEFUNCTIONS));
			exit(0);
		}
		file.close();
	}
	root=doc->documentElement();

	ClassNode classnode;
	QDomElement classroot=root.firstChildElement();
	while(!classroot.isNull())
	{
		classnode.load(classroot);
		classnodes.push_back(classnode);
		classroot=classroot.nextSiblingElement();
	}

	ruledoc=new QDomDocument(CREATERULE);
    filename=QString("%1/%2.xml").arg(ui.KernelDir->text()).arg(CREATERULE);
	file.setFileName(filename);
	if(!file.open(QIODevice::ReadOnly|QIODevice::Text))
	{
		ruledoc->appendChild(ruledoc->createElement(CREATERULE)).toElement();
	}
	else
	{
		if(!ruledoc->setContent(&file))
		{
			delete ruledoc;
			file.close();
			QMessageBox::information(this,QString("Error"),QString("%1.xml broken").arg(CREATERULE));
			exit(0);
		}
		file.close();
	}
	ruleroot=ruledoc->documentElement();

    showRule();
}

ScanInterfaceFunction::~ScanInterfaceFunction()
{
	if(doc!=NULL)
	{
		delete doc;
		doc=NULL;
	}
	if(ruledoc!=NULL)
	{
		delete ruledoc;
		ruledoc=NULL;
	}
}

void ScanInterfaceFunction::scanKernelDirSlot()
{
	if(doc!=NULL)
	{
		delete doc;
		doc=NULL;
	}
	doc=new QDomDocument(INTERFACEFUNCTIONS);
	doc->appendChild(doc->createElement(INTERFACEFUNCTIONS)).toElement();
	root=doc->documentElement();
	classnodes.clear();
    scan(ui.KernelDir->text());
    storeRule();
    showRule();
	QMessageBox::information(this,"Scan Kernel Dir","Finish Scanning");
}

void ScanInterfaceFunction::showInterfaceFunctionsSlot()
{
	ui.InterfaceFunctions->clear();
	QTreeWidgetItem * curitem=ui.ClassesFiles->currentItem();
	QString nodename=curitem->text(0);
	bool available=ruleroot.firstChildElement("ClassRule").firstChildElement(nodename).firstChildElement("Available").text().toInt()!=0;
	bool needinputports=ruleroot.firstChildElement("ClassRule").firstChildElement(nodename).firstChildElement("NeedInputPorts").text().toInt()!=0;
	bool needoutputports=ruleroot.firstChildElement("ClassRule").firstChildElement(nodename).firstChildElement("NeedOutputPorts").text().toInt()!=0;

	ui.ParamsHeader->setEnabled(available&&1);
	ui.ParamsType->setEnabled(available&&1);
	ui.InheritParams->setEnabled(available&&1);
	ui.BrowseParamsHeader->setEnabled(available&&1);
	ui.ClearParamsHeader->setEnabled(available&&1);

	ui.VarsHeader->setEnabled(available&&1);
	ui.VarsType->setEnabled(available&&1);
	ui.InheritVars->setEnabled(available&&1);
	ui.BrowseVarsHeader->setEnabled(available&&1);
	ui.ClearVarsHeader->setEnabled(available&&1);

	if(needinputports)
	{
        ui.InputParamsData->setEnabled(available&&1);
		ui.BrowseInputDataHeader->setEnabled(available&&1);
		ui.AddInputDataHeader->setEnabled(available&&1);
		ui.DeleteInputDataHeader->setEnabled(available&&1);
        ui.ClearInputDataHeader->setEnabled(available&&1);
		ui.InputDataheadersList->setEnabled(available&&1);
		ui.AddInputPort->setEnabled(available&&1);
		ui.DeleteInputPort->setEnabled(available&&1);
        ui.ClearInputPort->setEnabled(available&&1);
		ui.InputPortsSize->setEnabled(available&&1);
	}
	else
	{
        ui.InputParamsData->setEnabled(0);
		ui.BrowseInputDataHeader->setEnabled(0);
		ui.AddInputDataHeader->setEnabled(0);
		ui.DeleteInputDataHeader->setEnabled(0);
        ui.ClearInputDataHeader->setEnabled(0);
		ui.InputDataheadersList->setEnabled(0);
		ui.AddInputPort->setEnabled(0);
		ui.DeleteInputPort->setEnabled(0);
        ui.ClearInputPort->setEnabled(0);
		ui.InputPortsSize->setEnabled(0);
	}

	if(needoutputports)
	{
		ui.OutputPortsNumber->setEnabled(available&&1);
		ui.OutputDataHeader->setEnabled(available&&1);
		ui.OutputDataType->setEnabled(available&&1);
		ui.InheritOutputData->setEnabled(available&&1);
		ui.BrowseOutputDataHeader->setEnabled(available&&1);
		ui.ClearOutputDataHeader->setEnabled(available&&1);
	}
	else
	{
		ui.OutputPortsNumber->setEnabled(0);
		ui.OutputDataHeader->setEnabled(0);
		ui.OutputDataType->setEnabled(0);
		ui.InheritOutputData->setEnabled(0);
		ui.BrowseOutputDataHeader->setEnabled(0);
		ui.ClearOutputDataHeader->setEnabled(0);
	}

	bool exfuncflag=0;
	while(curitem!=NULL)
	{
		QString curnode=curitem->text(0);
		int i,n=classnodes.size();
		for(i=0;i<n;i++)
		{
			if(classnodes[i].classname==curnode)
			{
				QTreeWidgetItem * newitem=new QTreeWidgetItem(QStringList()<<classnodes[i].classname);
				ui.InterfaceFunctions->addTopLevelItem(newitem);
				int j,m=classnodes[i].interfacefunctions.size();
				for(j=0;j<m;j++)
				{
					bool extendable=ruleroot.firstChildElement("FuncRule").firstChildElement(classnodes[i].interfacefunctions[j].functionname).firstChildElement("FuncType").text().toInt()==3;
					exfuncflag|=extendable;
					if(extendable)
					{
						newitem->addChild(new QTreeWidgetItem(QStringList()<<QString()<<classnodes[i].interfacefunctions[j].functionname<<"Yes"));
					}
					else
					{
						newitem->addChild(new QTreeWidgetItem(QStringList()<<QString()<<classnodes[i].interfacefunctions[j].functionname<<"No"));
					}
				}
				break;
			}
		}
		curitem=curitem->parent();
	}

	if(exfuncflag)
	{
		ui.ExFunc->setEnabled(available&&1);
		ui.AddExFunc->setEnabled(available&&1);
		ui.DeleteExFunc->setEnabled(available&&1);
		ui.ClearExFuncs->setEnabled(available&&1);
		ui.OnlyExFunc->setEnabled(available&&1);
		ui.ExFuncs->setEnabled(available&&1);
	}
	else
	{
		ui.ExFunc->setEnabled(0);
		ui.AddExFunc->setEnabled(0);
		ui.DeleteExFunc->setEnabled(0);
		ui.ClearExFuncs->setEnabled(0);
		ui.OnlyExFunc->setEnabled(0);
		ui.ExFuncs->setEnabled(0);
	}

	ui.InterfaceFunctions->expandAll();
	ui.InterfaceFunctions->resizeColumnToContents(0);
	ui.InterfaceFunctions->resizeColumnToContents(1);
	ui.InterfaceFunctions->resizeColumnToContents(2);
}

void ScanInterfaceFunction::showDetailInformationSlot()
{
	ui.DetailInformation->clear();
	QTreeWidgetItem * curitem=ui.InterfaceFunctions->currentItem();
	if(curitem->text(0).size()>0)
	{
		return;
	}
	QTreeWidgetItem * paritem=curitem->parent();
	QString ownerclass=paritem->text(0);
	QString funcname=curitem->text(1);
	int i,n=classnodes.size();
	for(i=0;i<n;i++)
	{
		if(classnodes[i].classname==ownerclass)
		{
			
			int j,m=classnodes[i].interfacefunctions.size();
			for(j=0;j<m;j++)
			{
				if(classnodes[i].interfacefunctions[j].functionname==funcname)
				{
					QString functiondeclare=QString("%1 %2(").arg(classnodes[i].interfacefunctions[j].returnvalue).arg(classnodes[i].interfacefunctions[j].functionname);
					int k,l=classnodes[i].interfacefunctions[j].parameternames.size();
					for(k=0;k<l;k++)
					{
						functiondeclare=functiondeclare+QString("%1 %2, ").arg(classnodes[i].interfacefunctions[j].parametertypes[k]).arg(classnodes[i].interfacefunctions[j].parameternames[k]);
					}
					if(l>0)
					{
						int tmpsize=functiondeclare.size();
						functiondeclare.truncate(tmpsize-2);
					}
					functiondeclare=functiondeclare+QString(")");
					ui.DetailInformation->insertPlainText(functiondeclare);
					ui.DetailInformation->insertPlainText(QString("\n\nBelongs to class %1 in file <%2>.\n\n").arg(classnodes[i].classname).arg(classnodes[i].filename));
					ui.DetailInformation->insertPlainText(QString("In header, the function %1 will be declared as follow with some decorations (can be compiled in doxygen):\n\n").arg(classnodes[i].interfacefunctions[j].functionname));
					
					ui.DetailInformation->insertPlainText(QString("/*! \\fn %1\n").arg(functiondeclare));
					l=classnodes[i].interfacefunctions[j].comment.size();
					for(k=0;k<l;k++)
					{
						ui.DetailInformation->insertPlainText(QString("%1\n").arg(classnodes[i].interfacefunctions[j].comment[k]));
					}
					ui.DetailInformation->insertPlainText(QString("*/"));
				}
			}
		}
	}
}

void ScanInterfaceFunction::browseParamsHeaderSlot()
{
	QString headerfile=QFileDialog::getOpenFileName(this,"Add Params Header",ui.LibraryDir->text(),"ParamsData Header (*_ParamsData.h)");
	QString librarydir=ui.LibraryDir->text();
	if(headerfile.size()>0&&headerfile.startsWith(librarydir))
	{
		QList<QString> headerset;
		QList<QString> addedheader;
		headerset.push_back(headerfile);
		addedheader.push_back(headerfile);
		QStringList paramstypecandidate;
		QString paramstype;
		while(headerset.size()>0)
		{
			QString header=headerset.front();
			headerset.pop_front();
			QFile file(header);
			file.open(QIODevice::ReadOnly | QIODevice::Text);
			QTextStream textstream(&file);
			while(!textstream.atEnd())
			{
				QString tmpline=textstream.readLine().trimmed();
				if(tmpline.startsWith("class")&&tmpline.contains(QString(":")))
				{
					int tmpid=tmpline.indexOf(QString(":"));
					tmpline=tmpline.left(tmpid).trimmed();
				}
				if(tmpline.startsWith("class ")&&tmpline.endsWith("_Params"))
				{
					int startid=tmpline.indexOf(" ");
					paramstype=tmpline.mid(startid).remove("ROBOTSDK_OUTPUT").trimmed();
					paramstypecandidate<<paramstype;
				}
				else if(tmpline.startsWith("class ")&&tmpline.endsWith("_InputParams"))
				{
					int startid=tmpline.indexOf(" ");
					paramstype=tmpline.mid(startid).remove("ROBOTSDK_OUTPUT").trimmed();
					paramstypecandidate<<paramstype;
				}
				else if(tmpline.startsWith("#include<")&&tmpline.endsWith("_ParamsData.h>"))
				{
					int startid=tmpline.indexOf("<");
					int endid=tmpline.indexOf(">");
					QString headerfile=QString("%1\\%2").arg(librarydir).arg(tmpline.mid(startid+1,endid-startid-1));
					if(!addedheader.contains(headerfile))
					{
						addedheader.push_back(headerfile);
						headerset.push_back(headerfile);
					}
				}
			}
			file.close();			
		}
		headerfile.remove(QString("%1/").arg(librarydir));
		headerfile=QString("<%1>").arg(headerfile);
		paramstype=QInputDialog::getItem(this,QString("Choose Params Type"),QString("Params Type from %1").arg(headerfile),paramstypecandidate);
		if(paramstype.size()==0)
		{
			ui.ParamsHeader->clear();
			ui.ParamsType->clear();
			QMessageBox::information(this,"Error",QString("%1 doesn't contain _Params or _InputParams").arg(headerfile));
			return;
		}
		else
		{
			ui.ParamsHeader->setText(headerfile);
			ui.ParamsType->setText(paramstype);
		}
	}
}

void ScanInterfaceFunction::clearParamsHeaderSlot()
{
	ui.ParamsHeader->clear();
	ui.ParamsType->clear();
}

void ScanInterfaceFunction::browseOutputDataHeaderSlot()
{
	QString headerfile=QFileDialog::getOpenFileName(this,"Add Output Data Header",ui.LibraryDir->text(),"ParamsData Header (*_ParamsData.h)");
	QString librarydir=ui.LibraryDir->text();
	if(headerfile.size()>0&&headerfile.startsWith(librarydir))
	{
		QList<QString> headerset;
		QList<QString> addedheader;
		headerset.push_back(headerfile);
		addedheader.push_back(headerfile);
		QStringList outputdatatypecandidate;
		QString outputdatatype;
		while(headerset.size()>0)
		{
			QString header=headerset.front();
			headerset.pop_front();	
			QFile file(header);
			file.open(QIODevice::ReadOnly | QIODevice::Text);
			QTextStream textstream(&file);
			while(!textstream.atEnd())
			{
				QString tmpline=textstream.readLine().trimmed();
				if(tmpline.startsWith("class")&&tmpline.contains(QString(":")))
				{
					int tmpid=tmpline.indexOf(QString(":"));
					tmpline=tmpline.left(tmpid).trimmed();
				}
				if(tmpline.startsWith("class ")&&tmpline.endsWith("_Data"))
				{
					int startid=tmpline.indexOf(" ");
					outputdatatype=tmpline.mid(startid).remove("ROBOTSDK_OUTPUT").trimmed();
					outputdatatypecandidate<<outputdatatype;
				}
				else if(tmpline.startsWith("class ")&&tmpline.endsWith("_InputData"))
				{
					int startid=tmpline.indexOf(" ");
					outputdatatype=tmpline.mid(startid).remove("ROBOTSDK_OUTPUT").trimmed();
					outputdatatypecandidate<<outputdatatype;
				}
				else if(tmpline.startsWith("#include<")&&tmpline.endsWith("_ParamsData.h>"))
				{
					int startid=tmpline.indexOf("<");
					int endid=tmpline.indexOf(">");
                    QString headerfile=QString("%1/%2").arg(librarydir).arg(tmpline.mid(startid+1,endid-startid-1));
					if(!addedheader.contains(headerfile))
					{
						addedheader.push_back(headerfile);
						headerset.push_back(headerfile);
					}
				}
			}
			file.close();			
		}
    #ifdef Q_OS_LINUX
        if(headerfile.startsWith(ROBOTSDKMODULEDEV))
        {
            headerfile.remove(QString("%1/").arg(ROBOTSDKMODULEDEV));
        }
        else if(headerfile.startsWith(ROBOTSDKMODULE))
        {
            headerfile.remove(QString("%1/").arg(ROBOTSDKMODULE));
        }
    #elif defined(Q_OS_WIN)
        headerfile.remove(QString("%1/").arg(librarydir));
    #endif
		headerfile=QString("<%1>").arg(headerfile);
		outputdatatype=QInputDialog::getItem(this,QString("Choose Output Data Type"),QString("Output Data Type from %1").arg(headerfile),outputdatatypecandidate);
		if(outputdatatype.size()==0)
		{
			ui.OutputDataHeader->clear();
			ui.OutputDataType->clear();
			QMessageBox::information(this,"Error",QString("%1 doesn't contain _Data or _InputData").arg(headerfile));
			return;
		}
		else
		{
			ui.OutputDataHeader->setText(headerfile);
			ui.OutputDataType->setText(outputdatatype);
		}
	}
}

void ScanInterfaceFunction::clearOutputDataHeaderSlot()
{
	ui.OutputDataHeader->clear();
	ui.OutputDataType->clear();
}

void ScanInterfaceFunction::browseVarsHeaderSlot()
{
	QString headerfile=QFileDialog::getOpenFileName(this,"Add Vars Header",ui.LibraryDir->text(),"Vars Header (*_Vars.h)");
	QString librarydir=ui.LibraryDir->text();
	if(headerfile.size()>0&&headerfile.startsWith(librarydir))
	{
		QList<QString> headerset;
		QList<QString> addedheader;
		headerset.push_back(headerfile);
		addedheader.push_back(headerfile);
		QStringList varstypecandidate;
		QString varstype;
		while(headerset.size()>0)
		{
			QString header=headerset.front();
			headerset.pop_front();
			QFile file(header);
			file.open(QIODevice::ReadOnly | QIODevice::Text);
			QTextStream textstream(&file);
			while(!textstream.atEnd())
			{
				QString tmpline=textstream.readLine().trimmed();
				if(tmpline.startsWith("class")&&tmpline.contains(QString(":")))
				{
					int tmpid=tmpline.indexOf(QString(":"));
					tmpline=tmpline.left(tmpid).trimmed();
				}
				if(tmpline.startsWith("class ")&&tmpline.endsWith("_Vars"))
				{
					int startid=tmpline.indexOf(" ");
					varstype=tmpline.mid(startid).remove("ROBOTSDK_OUTPUT").trimmed();
					varstypecandidate<<varstype;
				}
				else if(tmpline.startsWith("#include<")&&tmpline.endsWith("_Vars.h>"))
				{
					int startid=tmpline.indexOf("<");
					int endid=tmpline.indexOf(">");
                    QString headerfile=QString("%1/%2").arg(librarydir).arg(tmpline.mid(startid+1,endid-startid-1));
					if(!addedheader.contains(headerfile))
					{
						addedheader.push_back(headerfile);
						headerset.push_back(headerfile);
					}
				}
			}
			file.close();			
		}
    #ifdef Q_OS_LINUX
        if(headerfile.startsWith(ROBOTSDKMODULEDEV))
        {
            headerfile.remove(QString("%1/").arg(ROBOTSDKMODULEDEV));
        }
        else if(headerfile.startsWith(ROBOTSDKMODULE))
        {
            headerfile.remove(QString("%1/").arg(ROBOTSDKMODULE));
        }
    #elif defined(Q_OS_WIN)
        headerfile.remove(QString("%1/").arg(librarydir));
    #endif
		headerfile=QString("<%1>").arg(headerfile);
		varstype=QInputDialog::getItem(this,QString("Choose Vars Type"),QString("Vars Type from %1").arg(headerfile),varstypecandidate);
		if(varstype.size()==0)
		{
			ui.VarsHeader->clear();
			ui.VarsType->clear();
			QMessageBox::information(this,"Error",QString("%1 doesn't contain _Vars").arg(headerfile));
			return;
		}
		else
		{
			ui.VarsHeader->setText(headerfile);
			ui.VarsType->setText(varstype);
		}
	}
}

void ScanInterfaceFunction::clearVarsHeaderSlot()
{
	ui.VarsHeader->clear();
	ui.VarsType->clear();
}

void ScanInterfaceFunction::addInputDataParamsHeadersSlot()
{
	QString inputparamsdataname=ui.InputParamsData->text();
	if(inputparamsdataname.size()==0)
	{
		return;
	}
	QString tmpinputparams=QString("%1_%2_%3_InputParams").arg(nodetypename).arg(nodeclassname).arg(inputparamsdataname);
	QString tmpinputdata=QString("%1_%2_%3_InputData").arg(nodetypename).arg(nodeclassname).arg(inputparamsdataname);
	if(ui.InputDataheadersList->findItems(tmpinputparams,Qt::MatchExactly).size()==0&&ui.InputDataheadersList->findItems(tmpinputdata,Qt::MatchExactly).size()==0)
	{
		ui.InputDataheadersList->setSortingEnabled(0);
		int rowid=ui.InputDataheadersList->rowCount();
		ui.InputDataheadersList->insertRow(rowid);
		ui.InputDataheadersList->setItem(rowid,0,new QTableWidgetItem("New Type"));
		nodetypename=ui.ClassesFiles->currentItem()->text(0);
		nodeclassname=ui.ClassName->text();
		ui.InputDataheadersList->setItem(rowid,1,new QTableWidgetItem(QString("%1_%2_%3_InputParams").arg(nodetypename).arg(nodeclassname).arg(inputparamsdataname)));
		ui.InputDataheadersList->setItem(rowid,2,new QTableWidgetItem(QString("%1_%2_%3_InputData").arg(nodetypename).arg(nodeclassname).arg(inputparamsdataname)));
		ui.InputDataheadersList->setCurrentCell(rowid,0);
		ui.InputDataheadersList->setSortingEnabled(1);
		ui.InputDataheadersList->sortByColumn(0,Qt::AscendingOrder);
		ui.InputParamsData->clear();
		ui.InputDataheadersList->resizeColumnsToContents();
	}
}

void ScanInterfaceFunction::browseInputDataParamsHeadersSlot()
{
	QStringList headers=QFileDialog::getOpenFileNames(this,"Add Input Data & Params Headers",ui.LibraryDir->text(),"ParamsData Header (*_ParamsData.h)");
	if(headers.size()>0)
	{
		QString librarydir=ui.LibraryDir->text();
		int i,n=headers.size();
		for(i=0;i<n;i++)
		{
			QString inputheader=headers.at(i);
			if(inputheader.startsWith(librarydir))
			{
				QList<QString> headerset;
				QList<QString> addedheader;
				headerset.push_back(inputheader);
				addedheader.push_back(inputheader);
				QStringList inputparamstypecandidate;
				QString inputparamstype;
				QStringList inputdatatypecandidate;
				QString inputdatatype;
				while(headerset.size()>0)
				{
					QString header=headerset.front();
					headerset.pop_front();
					QFile file(header);
					file.open(QIODevice::ReadOnly | QIODevice::Text);
					QTextStream textstream(&file);
					while(!textstream.atEnd())
					{
						QString tmpline=textstream.readLine().trimmed();
						if(tmpline.startsWith("class")&&tmpline.contains(QString(":")))
						{
							int tmpid=tmpline.indexOf(QString(":"));
							tmpline=tmpline.left(tmpid).trimmed();
						}
						if(tmpline.startsWith("class ")&&tmpline.endsWith("_Params"))
						{
							int startid=tmpline.indexOf(" ");
							inputparamstype=tmpline.mid(startid).remove("ROBOTSDK_OUTPUT").trimmed();
							inputparamstypecandidate<<inputparamstype;
						}
						else if(tmpline.startsWith("class ")&&tmpline.endsWith("_InputParams"))
						{
							int startid=tmpline.indexOf(" ");
							inputparamstype=tmpline.mid(startid).remove("ROBOTSDK_OUTPUT").trimmed();
							inputparamstypecandidate<<inputparamstype;
						}
						else if(tmpline.startsWith("class ")&&tmpline.endsWith("_Data"))
						{
							int startid=tmpline.indexOf(" ");
							inputdatatype=tmpline.mid(startid).remove("ROBOTSDK_OUTPUT").trimmed();
							inputdatatypecandidate<<inputdatatype;
						}
						else if(tmpline.startsWith("class ")&&tmpline.endsWith("_InputData"))
						{
							int startid=tmpline.indexOf(" ");
							inputdatatype=tmpline.mid(startid).remove("ROBOTSDK_OUTPUT").trimmed();
							inputdatatypecandidate<<inputdatatype;
						}
						else if(tmpline.startsWith("#include<")&&tmpline.endsWith("_ParamsData.h>"))
						{
							int startid=tmpline.indexOf("<");
							int endid=tmpline.indexOf(">");
                            QString headerfile=QString("%1/%2").arg(librarydir).arg(tmpline.mid(startid+1,endid-startid-1));
							if(!addedheader.contains(headerfile))
							{
								addedheader.push_back(headerfile);
								headerset.push_back(headerfile);
							}
						}
					}
					file.close();			
				}
            #ifdef Q_OS_LINUX
                if(inputheader.startsWith(ROBOTSDKMODULEDEV))
                {
                    inputheader.remove(QString("%1/").arg(ROBOTSDKMODULEDEV));
                }
                else if(inputheader.startsWith(ROBOTSDKMODULE))
                {
                    inputheader.remove(QString("%1/").arg(ROBOTSDKMODULE));
                }
            #elif defined(Q_OS_WIN)
                inputheader.remove(QString("%1/").arg(librarydir));
            #endif
				inputheader=QString("<%1>").arg(inputheader);
				inputparamstype=QInputDialog::getItem(this,QString("Choose Input Params Type"),QString("Input Params Type from %1").arg(inputheader),inputparamstypecandidate);
				inputdatatype=QInputDialog::getItem(this,QString("Choose Input Data Type"),QString("Input Data Type from %1").arg(inputheader),inputdatatypecandidate);
				if(inputparamstype.size()==0||inputdatatype.size()==0)
				{
					QMessageBox::information(this,"Error",QString("%1 doesn't contain Input Params and Input Data").arg(inputheader));
					return;
				}
				else
				{
					ui.InputDataheadersList->setSortingEnabled(0);
					int rowid=ui.InputDataheadersList->rowCount();
					ui.InputDataheadersList->insertRow(rowid);
					ui.InputDataheadersList->setItem(rowid,0,new QTableWidgetItem(inputheader));
					ui.InputDataheadersList->setItem(rowid,1,new QTableWidgetItem(inputparamstype));
					ui.InputDataheadersList->setItem(rowid,2,new QTableWidgetItem(inputdatatype));
					ui.InputDataheadersList->setCurrentCell(rowid,0);
					ui.InputDataheadersList->setSortingEnabled(1);
					ui.InputDataheadersList->sortByColumn(0,Qt::AscendingOrder);
				}
			}
		}
		ui.InputDataheadersList->resizeColumnsToContents();
	}
}

void ScanInterfaceFunction::deleteInputDataParamsHeadersSlot()
{
	int index=ui.InputDataheadersList->currentRow();
	if(index>=0)
	{
		ui.InputDataheadersList->removeRow(index);
		ui.InputDataheadersList->resizeColumnsToContents();
	}
}

void ScanInterfaceFunction::clearInputDataParamsHeadersSlot()
{
    ui.InputDataheadersList->clear();
    ui.InputDataheadersList->setRowCount(0);
}

void ScanInterfaceFunction::addInputPortsSizeSlot()
{
	int rowid=ui.InputPortsSize->rowCount();
	ui.InputPortsSize->insertRow(rowid);
	ui.InputPortsSize->setVerticalHeaderItem(rowid,new QTableWidgetItem(QString("%1").arg(rowid)));
	ui.InputPortsSize->setItem(rowid,0,new QTableWidgetItem(QString("0")));
	ui.InputPortsSize->setItem(rowid,1,new QTableWidgetItem(QString()));
	ui.InputPortsSize->setItem(rowid,2,new QTableWidgetItem(QString()));
	ui.InputPortsSize->setItem(rowid,3,new QTableWidgetItem(QString()));
	ui.InputPortsSize->setCurrentCell(rowid,0);
	ui.InputPortsSize->resizeColumnsToContents();
}

void ScanInterfaceFunction::deleteInputPortsSizeSlot()
{
	int index=ui.InputPortsSize->currentRow();
	if(index>=0)
	{
		ui.InputPortsSize->removeRow(index);
		ui.InputPortsSize->resizeColumnsToContents();
	}
}

void ScanInterfaceFunction::clearInputPortsSizeSlot()
{
    ui.InputPortsSize->clear();
    ui.InputDataheadersList->setRowCount(0);
}

void ScanInterfaceFunction::processDoubleClickInputParamsDataSlot(int row, int column)
{
	QString librarydir=ui.LibraryDir->text();
	QString headerfile=ui.InputDataheadersList->item(row,0)->text();
	headerfile.remove("<");
	headerfile.remove(">");
    headerfile=QString("%1/%2").arg(librarydir).arg(headerfile);
	switch (column)
	{
	case 0:
		break;
	case 1:
		{
			QList<QString> headerset;
			QList<QString> addedheader;
			headerset.push_back(headerfile);
			addedheader.push_back(headerfile);
			QStringList inputparamstypecandidate;
			QString inputparamstype;
			while(headerset.size()>0)
			{
				QString header=headerset.front();
				headerset.pop_front();
				QFile file(header);
				file.open(QIODevice::ReadOnly | QIODevice::Text);
				QTextStream textstream(&file);
				while(!textstream.atEnd())
				{
					QString tmpline=textstream.readLine().trimmed();
					if(tmpline.startsWith("class ")&&tmpline.endsWith("_Params"))
					{
						int startid=tmpline.indexOf(" ");
						inputparamstype=tmpline.mid(startid).remove("ROBOTSDK_OUTPUT").trimmed();
						inputparamstypecandidate<<inputparamstype;
					}
					else if(tmpline.startsWith("class ")&&tmpline.endsWith("_InputParams"))
					{
						int startid=tmpline.indexOf(" ");
						inputparamstype=tmpline.mid(startid).remove("ROBOTSDK_OUTPUT").trimmed();
						inputparamstypecandidate<<inputparamstype;
					}
					else if(tmpline.startsWith("#include<")&&tmpline.endsWith("_ParamsData.h>"))
					{
						int startid=tmpline.indexOf("<");
						int endid=tmpline.indexOf(">");
                        QString headerfile=QString("%1/%2").arg(librarydir).arg(tmpline.mid(startid+1,endid-startid-1));
						if(!addedheader.contains(headerfile))
						{
							addedheader.push_back(headerfile);
							headerset.push_back(headerfile);
						}
					}
				}
				file.close();			
			}
			headerfile=ui.InputDataheadersList->item(row,0)->text();
			inputparamstype=QInputDialog::getItem(this,"Choose Input Params Type","Input Params Type",inputparamstypecandidate);
			if(inputparamstype.size()>0)
			{
				ui.InputDataheadersList->item(row,column)->setText(inputparamstype);
			}
		}
		break;
	case 2:
		{
			QList<QString> headerset;
			QList<QString> addedheader;
			headerset.push_back(headerfile);
			addedheader.push_back(headerfile);
			QStringList inputdatatypecandidate;
			QString inputdatatype;
			while(headerset.size()>0)
			{
				QString header=headerset.front();
				headerset.pop_front();
				QFile file(header);
				file.open(QIODevice::ReadOnly | QIODevice::Text);
				QTextStream textstream(&file);
				while(!textstream.atEnd())
				{
					QString tmpline=textstream.readLine().trimmed();
					if(tmpline.startsWith("class ")&&tmpline.endsWith("_Params"))
					{
						int startid=tmpline.indexOf(" ");
						inputdatatype=tmpline.mid(startid).remove("ROBOTSDK_OUTPUT").trimmed();
						inputdatatypecandidate<<inputdatatype;
					}
					else if(tmpline.startsWith("class ")&&tmpline.endsWith("_InputParams"))
					{
						int startid=tmpline.indexOf(" ");
						inputdatatype=tmpline.mid(startid).remove("ROBOTSDK_OUTPUT").trimmed();
						inputdatatypecandidate<<inputdatatype;
					}
					else if(tmpline.startsWith("#include<")&&tmpline.endsWith("_ParamsData.h>"))
					{
						int startid=tmpline.indexOf("<");
						int endid=tmpline.indexOf(">");
                        QString headerfile=QString("%1/%2").arg(librarydir).arg(tmpline.mid(startid+1,endid-startid-1));
						if(!addedheader.contains(headerfile))
						{
							addedheader.push_back(headerfile);
							headerset.push_back(headerfile);
						}
					}
				}
				file.close();			
			}
			headerfile=ui.InputDataheadersList->item(row,0)->text();
			inputdatatype=QInputDialog::getItem(this,"Choose Input Data Type","Input Data Type",inputdatatypecandidate);
			if(inputdatatype.size()>0)
			{
				ui.InputDataheadersList->item(row,column)->setText(inputdatatype);
			}
		}
		break;
	default:
		break;
	}
}

void ScanInterfaceFunction::processDoubleClickInputPortSizeSlot(int row, int column)
{
	switch(column)
	{
	case 0:
		{
			int buffersize=QInputDialog::getInt(this,QString("Set Data Buffer Size of Input Port"),QString("Input Port %1").arg(row),0,0);
			ui.InputPortsSize->item(row,column)->setText(QString("%1").arg(buffersize));
			ui.InputPortsSize->resizeColumnsToContents();
		}
		break;
	case 1:
		{			
			int i,n=ui.InputDataheadersList->rowCount();
			if(n==0)
			{
				break;
			}
			QStringList headerlist;
			for(i=0;i<n;i++)
			{
				if(ui.InputDataheadersList->item(i,0)->text()!=QString("New Type"))
				{
					headerlist<<ui.InputDataheadersList->item(i,0)->text();
				}
			}
			if(headerlist.size()>0)
			{
				QString header=QInputDialog::getItem(this,QString("Set Input Params & Data Header for Input Port"),QString("Input Port %1").arg(row),headerlist,0,0);
				if(header.size()>0)
				{
					int rowid=ui.InputDataheadersList->findItems(header,Qt::MatchExactly).front()->row();
					ui.InputPortsSize->item(row,1)->setText(ui.InputDataheadersList->item(rowid,0)->text());					
					ui.InputPortsSize->item(row,2)->setText(ui.InputDataheadersList->item(rowid,1)->text());
					ui.InputPortsSize->item(row,3)->setText(ui.InputDataheadersList->item(rowid,2)->text());
					ui.InputPortsSize->resizeColumnsToContents();
				}
			}
		}
		break;
	case 2:
		{
			int i,n=ui.InputDataheadersList->rowCount();
			if(n==0)
			{
				break;
			}
			QStringList paramslist;
			for(i=0;i<n;i++)
			{
				paramslist<<ui.InputDataheadersList->item(i,1)->text();
			}
			if(paramslist.size()>0)
			{
				QString params=QInputDialog::getItem(this,QString("Set Input Params Type"),QString("Input Port %1").arg(row),paramslist,0,0);
				if(params.size()>0)
				{
					int rowid=ui.InputDataheadersList->findItems(params,Qt::MatchExactly).front()->row();
					ui.InputPortsSize->item(row,1)->setText(ui.InputDataheadersList->item(rowid,0)->text());					
					ui.InputPortsSize->item(row,2)->setText(ui.InputDataheadersList->item(rowid,1)->text());
					ui.InputPortsSize->item(row,3)->setText(ui.InputDataheadersList->item(rowid,2)->text());
					ui.InputPortsSize->resizeColumnsToContents();
				}
			}
		}
		break;
	case 3:
		{
			int i,n=ui.InputDataheadersList->rowCount();
			if(n==0)
			{
				break;
			}
			QStringList datalist;
			for(i=0;i<n;i++)
			{
				datalist<<ui.InputDataheadersList->item(i,2)->text();
			}
			if(datalist.size()>0)
			{
				QString data=QInputDialog::getItem(this,QString("Set Input Data Type"),QString("Input Port %1").arg(row),datalist,0,0);
				if(data.size()>0)
				{
					int rowid=ui.InputDataheadersList->findItems(data,Qt::MatchExactly).front()->row();
					ui.InputPortsSize->item(row,1)->setText(ui.InputDataheadersList->item(rowid,0)->text());					
					ui.InputPortsSize->item(row,2)->setText(ui.InputDataheadersList->item(rowid,1)->text());
					ui.InputPortsSize->item(row,3)->setText(ui.InputDataheadersList->item(rowid,2)->text());
					ui.InputPortsSize->resizeColumnsToContents();
				}
			}
		}
		break;
	}
}

void ScanInterfaceFunction::addExFuncSlot()
{
	QString tmpexfunc=ui.ExFunc->text();
	if(tmpexfunc.size()>0&&ui.ExFuncs->findItems(tmpexfunc,Qt::MatchExactly).size()==0)
	{
		ui.ExFuncs->insertItem(ui.ExFuncs->count(),tmpexfunc);
	}
}

void ScanInterfaceFunction::deleteExFuncSlot()
{
	int index=ui.ExFuncs->currentRow();
	if(index>=0)
	{
		delete(ui.ExFuncs->takeItem(index));
	}
}

void ScanInterfaceFunction::clearExFuncsSlot()
{
	ui.ExFuncs->clear();
}

void ScanInterfaceFunction::browseOutputDirSlot()
{
	QString vcxproj;
	if (ui.newqtpro->isChecked())
	{
		vcxproj = QFileDialog::getSaveFileName(this,"Output Qt Project",QString("%1\\").arg(ui.LibraryDir->text()),QString("Qt Project File (*.pro)"));
	}
	else
	{
		vcxproj = QFileDialog::getOpenFileName(this, "Output Project", ui.LibraryDir->text(), QString("Project File (*.vcxproj *.pro)"));
	}    
	if(vcxproj.size()>0)
	{
		if(vcxproj.startsWith(ui.LibraryDir->text()))
		{
			ui.OutputDir->setText(vcxproj);
		}
        else
        {
            QMessageBox::information(this,"Error","You can only output to the project in ModuleDev folder!");
        }
	}
}

void ScanInterfaceFunction::createLibraryFilesSlot()
{
    if(createRule())
	{
		QMessageBox::information(this,"Create Shared Library Files","Finish Creating");
	}
}

void ScanInterfaceFunction::scan(QString directory)
{
	QDir dir(directory);
	if(!dir.exists())
	{
		return;
	}
	dir.setFilter(QDir::Dirs | QDir::Files | QDir::NoDotAndDotDot);
	dir.setSorting(QDir::DirsFirst);
	QFileInfoList list=dir.entryInfoList();
	int i,n=list.size();
	for(i=0;i<n;i++)
	{
		QFileInfo fileinfo=list.at(i);
		QString filename=fileinfo.absoluteFilePath();
		if(fileinfo.isDir())
		{
			scan(filename);
		}
		else if(fileinfo.fileName().endsWith(".h"))
		{
			ClassNode classnode;
			QFile file(filename);
			file.open(QIODevice::ReadOnly | QIODevice::Text);
			QTextStream textstream(&file);
			QString relativefilepath=filename;
            relativefilepath.remove(QString("%1/").arg(QString("%1/include").arg(ROBOTSDKKERNEL)));
			while(classnode.scan(textstream,relativefilepath))
			{
				classnodes.push_back(classnode);
				classnode.store(doc,root);
				checkRules(classnode);
			}
		}
	}
}

void ScanInterfaceFunction::checkRules(ClassNode & classNode)
{
	QDomElement classrule=checkTag(ruleroot,"ClassRule");
	classrule=checkTag(classrule,classNode.classname);
	{	
		checkTag(classrule,"Available","1");
		checkTag(classrule,"NeedInputPorts","0");
		checkTag(classrule,"NeedOutputPorts","0");
	}

	QDomElement funcrule=checkTag(ruleroot,"FuncRule");
	int i,n=classNode.interfacefunctions.size();
	for(i=0;i<n;i++)
	{
		QDomElement funcelem=checkTag(funcrule,classNode.interfacefunctions[i].functionname);
		funcelem.setAttribute("Return",classNode.interfacefunctions[i].returnvalue);
		checkTag(funcelem,"FuncType","1");
		QDomElement paramselem=checkTag(funcelem,"Parameters");
		int j,m=classNode.interfacefunctions[i].parameternames.size();
		for(j=0;j<m;j++)
		{
			QDomElement paramelem=checkTag(paramselem,classNode.interfacefunctions[i].parameternames[j]);
			if(!paramelem.hasChildNodes())
			{
				paramelem.setAttribute("Type",classNode.interfacefunctions[i].parametertypes[j]);
				paramelem.appendChild(doc->createTextNode(QString()));
			}
		}
		QDomElement funcpreset=checkTag(funcelem,"FuncPreset");
		if(!funcpreset.hasChildNodes())
		{
			funcpreset.appendChild(doc->createTextNode(QString()));
		}
	}
}

QDomElement ScanInterfaceFunction::checkTag(QDomElement & tmproot, QString tag, QString defaultValue)
{
	QDomElement curelem=tmproot.firstChildElement(tag);
	if(curelem.isNull())
	{
		curelem=tmproot.appendChild(doc->createElement(tag)).toElement();
		if(defaultValue.size()>0)
		{
			curelem.appendChild(doc->createTextNode(defaultValue));
		}
	}
	return curelem;
}

void ScanInterfaceFunction::showRule()
{
	ui.ClassesFiles->clear();
	QList<QTreeWidgetItem *> queue;
	QTreeWidgetItem * item=new QTreeWidgetItem(QStringList()<<"QObject"<<"No"<<"<qobject.h>");
	ui.ClassesFiles->addTopLevelItem(item);
	queue.push_back(item);
	while(!queue.isEmpty())
	{
		item=queue.front();
		queue.pop_front();
		QString parentnode=item->text(0);
		int i,n=classnodes.size();
		for(i=0;i<n;i++)
		{
			if(classnodes[i].parentclass==parentnode)
			{
				QTreeWidgetItem * childitem;
				if(ruleroot.firstChildElement("ClassRule").firstChildElement(classnodes[i].classname).firstChildElement("Available").text().toInt()==0)
				{
					childitem=new QTreeWidgetItem(QStringList()<<classnodes[i].classname<<QString("No")<<QString("<%1>").arg(classnodes[i].filename));
				}
				else
				{
					childitem=new QTreeWidgetItem(QStringList()<<classnodes[i].classname<<QString("Yes")<<QString("<%1>").arg(classnodes[i].filename));
				}
				item->addChild(childitem);
				queue.push_back(childitem);
			}
		}
	}
	ui.ClassesFiles->expandAll();
	ui.ClassesFiles->resizeColumnToContents(0);
	ui.ClassesFiles->resizeColumnToContents(1);
	ui.ClassesFiles->resizeColumnToContents(2);

    ui.InputParamsData->setEnabled(0);
    ui.BrowseInputDataHeader->setEnabled(0);
    ui.AddInputDataHeader->setEnabled(0);
    ui.DeleteInputDataHeader->setEnabled(0);
    ui.ClearInputDataHeader->setEnabled(0);
    ui.InputDataheadersList->setEnabled(0);
    ui.AddInputPort->setEnabled(0);
    ui.DeleteInputPort->setEnabled(0);
    ui.ClearInputPort->setEnabled(0);
    ui.InputPortsSize->setEnabled(0);
	
	ui.OutputPortsNumber->setEnabled(0);

	ui.OutputDataHeader->setEnabled(0);
	ui.OutputDataType->setEnabled(0);
	ui.InheritOutputData->setEnabled(0);
	ui.BrowseOutputDataHeader->setEnabled(0);
	ui.ClearOutputDataHeader->setEnabled(0);

	ui.ParamsHeader->setEnabled(0);
	ui.ParamsType->setEnabled(0);
	ui.InheritParams->setEnabled(0);
	ui.BrowseParamsHeader->setEnabled(0);
	ui.ClearParamsHeader->setEnabled(0);

	ui.VarsHeader->setEnabled(0);
	ui.VarsType->setEnabled(0);
	ui.InheritVars->setEnabled(0);
	ui.BrowseVarsHeader->setEnabled(0);
	ui.ClearVarsHeader->setEnabled(0);
	
	ui.ExFunc->setEnabled(0);
	ui.AddExFunc->setEnabled(0);
	ui.DeleteExFunc->setEnabled(0);
	ui.ClearExFuncs->setEnabled(0);
	ui.OnlyExFunc->setEnabled(0);
	ui.ExFuncs->setEnabled(0);
}

void ScanInterfaceFunction::storeRule()
{
    QString filename=QString("%1/%2.xml").arg(ui.KernelDir->text()).arg(INTERFACEFUNCTIONS);
	QFile file(filename);
	file.open(QIODevice::WriteOnly | QIODevice::Text);
	QTextStream textstream(&file);
	doc->save(textstream,2);
	file.close();

    filename=QString("%1/%2.h").arg(ui.KernelDir->text()).arg(ROBOTSDKGLOBAL);
	file.setFileName(filename);
	file.open(QIODevice::WriteOnly | QIODevice::Text);
	textstream.setDevice(&file);

	textstream<<"#ifndef ROBOTSDK_GLOBAL_H\n";
	textstream<<"#define ROBOTSDK_GLOBAL_H\n\n";

	textstream<<QString("/*! \\addtogroup Kernel\n").arg(nodetypename).arg(nodeclassname);
	textstream<<QString("\t@{\n");
	textstream<<QString("*/\n\n");

	textstream<<QString("/*! \\file %1.h\n").arg(ROBOTSDKGLOBAL);
	textstream<<QString("\t\\brief RobotSDK global definition.\n");
	textstream<<QString("\t\\details\n");
	textstream<<QString("\t- Used in shared library creation.\n");
	textstream<<QString("\t- Used in application's reference to shared library's _ParamsData.h.\n");
	textstream<<QString("*/\n\n");

	textstream<<"#include<qglobal.h>\n\n";

	textstream<<QString("/*! \\def ROBOTSDK_OUTPUT\n");
    textstream<<QString("\t\\brief Defines the output type according to the Macro RobotSDK_ModuleDev in the project.\n");
	textstream<<QString("\t\\details\n");
    textstream<<QString("\t- If undefined RobotSDK_ModuleDev in the project, output type=Q_DECL_IMPORT which is for application.\n");
    textstream<<QString("\t- If defined RobotSDK_ModuleDev in the project, output type=Q_DECL_EXPORT which is for shared library.\n");
	textstream<<QString("*/\n");
    textstream<<"#ifdef RobotSDK_ModuleDev\n";
    textstream<<"#define ROBOTSDK_OUTPUT Q_DECL_EXPORT\n";
	textstream<<"#else\n";
	textstream<<"#define ROBOTSDK_OUTPUT Q_DECL_IMPORT\n";
	textstream<<"#endif\n\n";

	textstream<<QString("/*! \\def DECOFUNC(func)\n");
	textstream<<QString("\t\\brief To decorate function \a func.\n");
	textstream<<QString("*/\n");
	textstream<<QString("/*! \\def DECOFUNC_1(NODECONFIG,func)\n");
	textstream<<QString("\t\\brief To decorate function \a func using NODECONFIG.\n");
	textstream<<QString("*/\n");
	textstream<<QString("/*! \\def DECOFUNC_2(NODECONFIG,func)\n");
	textstream<<QString("\t\\brief To decorate function \a func using NODECONFIG and get function name as NODECONFIG_func.\n");
	textstream<<QString("*/\n");
	textstream<<"#define DECOFUNC(func) DECOFUNC_1(NODECONFIG,func)\n";
	textstream<<"#define DECOFUNC_1(NODECONFIG,func) DECOFUNC_2(NODECONFIG,func)\n";
	textstream<<"#define DECOFUNC_2(NODECONFIG,func) NODECONFIG##_##func\n\n";

	QSet<QString> tmpheaders;
	int i,n=classnodes.size();
	for(i=0;i<n;i++)
	{
		tmpheaders.insert(QString("#include<%1>\n").arg(classnodes[i].filename));
	}
	QStringList headers=tmpheaders.toList();
	n=headers.size();
	for(i=0;i<n;i++)
	{
		textstream<<headers.at(i);
	}
	textstream<<"\n";

	textstream<<"#include<qglobal.h>\n";
	textstream<<"#include<qdebug.h>\n";
    textstream<<"#include<qlabel.h>\n";
    textstream<<"#include<qlineedit.h>\n";
	textstream<<"#include<qstring.h>\n";
	textstream<<"#include<qfile.h>\n";
	textstream<<"#include<qlist.h>\n";
	textstream<<"#include<qvector.h>\n";
	textstream<<"#include<qset.h>\n";
	textstream<<"#include<qfile.h>\n";
	textstream<<"#include<qtextstream.h>\n";
	textstream<<"#include<qdatetime.h>\n";
	textstream<<"#include<qtimer.h>\n";
	textstream<<"#include<qimage.h>\n";
	textstream<<"#include<qpainter.h>\n";
	textstream<<"#include<qrgb.h>\n";
    textstream<<"#include<boost/shared_ptr.hpp>\n";
    textstream<<"#include<Accessories/XMLDomInterface/xmldominterface.h>\n";
	textstream<<"\n";

	/*! \fn QVector<boost::shared_ptr<void> > grabInputParams(int grabSize)
        \brief Grab the input parameters in the size of \a grabSize from InputPort::inputparamsbuffer.
        \param [in] grabSize The size of grabbed input parameters.
        \return The grabbed input parameters in QVector.
    */
	textstream<<"/*!\t\\fn void copyQVector(QVector<T1 *> & dst, QVector<T2 *> & src)\n";
	textstream<<"\t\\brief Copy and convert pointers.\n";
	textstream<<"\t\\param [in] dst The destination to store pointers.\n";
	textstream<<"\t\\param [in] src The source to copy.\n";
	textstream<<"*/\n";

	textstream<<"template<class T1, class T2>\n";
	textstream<<"void copyQVector(QVector<T1 *> & dst, QVector<T2 *> & src)\n";
	textstream<<"{\n";
	textstream<<"\tint i,n=src.size();\n";
	textstream<<"\tdst.resize(n);\n";
	textstream<<"\tfor(i=0;i<n;i++)\n";
	textstream<<"\t{\n";
	textstream<<"\t\tdst[i]=(T1 *)src[i];\n";
	textstream<<"\t}\n";
	textstream<<"}\n";
	textstream<<"\n";

	textstream<<QString("/*! @}*/ \n\n");

	textstream<<"#endif";

	file.close();

    filename=QString("%1/%2.xml").arg(ui.KernelDir->text()).arg(CREATERULE);
	file.setFileName(filename);
	file.open(QIODevice::WriteOnly | QIODevice::Text);
	textstream.setDevice(&file);
	ruledoc->save(textstream,2);
	file.close();
}

bool ScanInterfaceFunction::createRule()
{
	if(!readConfiguration())
	{
		return 0;
	}

	createFiles();
    QString projname=ui.OutputDir->text();
    if(projname.endsWith(".pro"))
    {
        configQtProject();
    }
    else if(projname.endsWith(".vcxproj"))
    {
        configVSProject();
    }
	return 1;
}

bool ScanInterfaceFunction::readConfiguration()
{
	QTreeWidgetItem * curitem=ui.ClassesFiles->currentItem();
	if(curitem==NULL||curitem->text(0)=="QObject")
	{
		QMessageBox::information(this,"Error",QString("Please select an available class for node's type-name.").arg(nodetypename));
		return 0;
	}
	nodetypename=curitem->text(0);
	if(ruleroot.firstChildElement("ClassRule").firstChildElement(nodetypename).firstChildElement("Available").text().toInt()==0)
	{
		QMessageBox::information(this,"Error",QString("%1 is not available for library creation.").arg(nodetypename));
		return 0;
	}

	bool needinputports=ruleroot.firstChildElement("ClassRule").firstChildElement(nodetypename).firstChildElement("NeedInputPorts").text().toInt()!=0;
	bool needoutputports=ruleroot.firstChildElement("ClassRule").firstChildElement(nodetypename).firstChildElement("NeedOutputPorts").text().toInt()!=0;

	if(needinputports)
	{
		if(ui.InputPortsSize->rowCount()==0)
		{
			QMessageBox::information(this,"Error",QString("%1 needs input ports.").arg(nodetypename));
			return 0;
		}
	}

	if(needoutputports)
	{
		if(ui.OutputPortsNumber->value()==0)
		{
			QMessageBox::information(this,"Error",QString("%1 needs output ports.").arg(nodetypename));
			return 0;
		}
	}

	interfacefunctions.clear();
	QString tmpnodetypename=nodetypename;
	while(tmpnodetypename!="QObject")
	{
		int i,n=classnodes.size();
		for(i=0;i<n;i++)
		{
			if(classnodes[i].classname==tmpnodetypename)
			{
				interfacefunctions=classnodes[i].interfacefunctions+interfacefunctions;
				tmpnodetypename=classnodes[i].parentclass;
				break;
			}
		}
		if(i==n)
		{
			break;
		}
	}
	if(interfacefunctions.size()==0)
	{
		QMessageBox::information(this,"Error",QString("There is no interface functions need to be packed in shared library."));
		return 0;
	}

	bool exfuncflag=0;
	int i,n=interfacefunctions.size();
	for(i=0;i<n;i++)
	{
		exfuncflag|=ruleroot.firstChildElement("FuncRule").firstChildElement(interfacefunctions[i].functionname).firstChildElement("FuncType").text().toInt()==3;
	}

	nodeclassname=ui.ClassName->text();
	if(nodeclassname.size()==0)
	{
		QMessageBox::information(this,"Error",QString("Need to name node's class-name."));
		return 0;
	}

	paramsheader=ui.ParamsHeader->text();
	paramstype=ui.ParamsType->text();
	outputdataheader=ui.OutputDataHeader->text();
	outputdatatype=ui.OutputDataType->text();
	varsheader=ui.VarsHeader->text();
	varstype=ui.VarsType->text();

	inputheaders.clear();
	inputparamstypes.clear();
	inputdatatypes.clear();
	inputportssize.clear();
	inputportsparamstypes.clear();
	inputportsdatatypes.clear();
	if(needinputports)
	{
		n=ui.InputDataheadersList->rowCount();
		for(i=0;i<n;i++)
		{
			inputheaders<<ui.InputDataheadersList->item(i,0)->text();
			inputparamstypes<<ui.InputDataheadersList->item(i,1)->text();
			inputdatatypes<<ui.InputDataheadersList->item(i,2)->text();
		}	
		n=ui.InputPortsSize->rowCount();
		for(i=0;i<n;i++)
		{
			inputportssize<<ui.InputPortsSize->item(i,0)->text().toInt();
			inputportsparamstypes<<ui.InputPortsSize->item(i,2)->text();
			inputportsdatatypes<<ui.InputPortsSize->item(i,3)->text();
		}
	}

	if(needoutputports)
	{
		outputportsnumber=ui.OutputPortsNumber->value();
	}
	else
	{
		outputportsnumber=0;
	}

	exfuncs.clear();
	if(exfuncflag)
	{
		n=ui.ExFuncs->count();
		for(i=0;i<n;i++)
		{
			exfuncs<<ui.ExFuncs->item(i)->text();
		}
	}

	outputvcxproj=ui.OutputDir->text();
	if(outputvcxproj.size()==0)
	{
		QMessageBox::information(this,"Error",QString("Need to give output dir."));
		return 0;
	}
	int tmpid=outputvcxproj.lastIndexOf("/");
	outputdir=outputvcxproj.left(tmpid);
	return 1;
}

void ScanInterfaceFunction::createFiles()
{
    QString editpath=nodeclassname;
    QString noeditpath=nodeclassname;
    {
        editpath=QString("%1/%2/%3/%4").arg(outputdir).arg(editpath.replace("_","/")).arg(nodetypename).arg(EDITFOLDER);
        QDir dir(editpath);
        if(!dir.exists())
        {
            dir.mkpath(editpath);
        }
    }
    {
        noeditpath=QString("%1/%2/%3/%4").arg(outputdir).arg(noeditpath.replace("_","/")).arg(nodetypename).arg(NOEDITFOLDER);
        QDir dir(noeditpath);
        if(!dir.exists())
        {
            dir.mkpath(noeditpath);
        }
    }

    QString paramsdata_h=QString("%1/%2_%3_ParamsData.h").arg(editpath).arg(nodetypename).arg(nodeclassname);
    QString vars_h=QString("%1/%2_%3_Vars.h").arg(editpath).arg(nodetypename).arg(nodeclassname);
    QString privatefunc_h=QString("%1/%2_%3_PrivFunc.h").arg(noeditpath).arg(nodetypename).arg(nodeclassname);
    QString privatefunc_cpp=QString("%1/%2_%3_PrivFunc.cpp").arg(editpath).arg(nodetypename).arg(nodeclassname);
    QString privatecorefunc_h=QString("%1/%2_%3_PrivCoreFunc.h").arg(noeditpath).arg(nodetypename).arg(nodeclassname);
    QString privatecorefunc_cpp=QString("%1/%2_%3_PrivCoreFunc.cpp").arg(noeditpath).arg(nodetypename).arg(nodeclassname);
    QString privateexfunc_h=QString("%1/%2_%3_PrivExFunc.h").arg(noeditpath).arg(nodetypename).arg(nodeclassname);
    QString privateexfunc_cpp=QString("%1/%2_%3_PrivExFunc.cpp").arg(editpath).arg(nodetypename).arg(nodeclassname);

	QFile file;
	QTextStream textstream;
	
	if(!ui.OnlyExFunc->isChecked())
	{
		file.setFileName(paramsdata_h);
		if(file.open(QIODevice::WriteOnly | QIODevice::Text))
		{
			textstream.setDevice(&file);

			textstream<<QString("//You need to modify this file.\n\n");

			writeHead(textstream,QString("ParamsData"));

			if(inputheaders.size()>0)
			{
				int i,n=inputheaders.size();
				int countheader=0;
				for(i=0;i<n;i++)
				{
					if(inputheaders.at(i)!=QString("New Type"))
					{
						countheader++;
					}
				}
				textstream<<QString("//%1 input data header(s) refered\n\n").arg(countheader);
				for(i=0;i<n;i++)
				{
					if(inputheaders.at(i)!=QString("New Type"))
					{
						textstream<<QString("//Defines Params %1 and Input Data %2\n").arg(inputparamstypes.at(i)).arg(inputdatatypes.at(i));
						textstream<<QString("#include%1\n").arg(inputheaders.at(i));
					}
				}
				textstream<<QString("\n");
				textstream<<QString("//%1 new input data type(s) created\n\n").arg(inputheaders.size()-countheader);
				QString tmpheader=QString("%1_%2_").arg(nodetypename).arg(nodeclassname);
				for(i=0;i<n;i++)
				{
					if(inputheaders.at(i)==QString("New Type"))
					{
						QString tmpinputparamstype=inputparamstypes.at(i);
						tmpinputparamstype.remove(tmpheader);
						writeClass(textstream,tmpinputparamstype);
						QString tmpinputdatatype=inputdatatypes.at(i);
						tmpinputdatatype.remove(tmpheader);
						writeClass(textstream,tmpinputdatatype);
					}
				}

				n=inputportssize.size();
				QString tmpsize=QString("QList<int>()");
				for(i=0;i<n;i++)
				{
					tmpsize=tmpsize+QString("<<%1").arg(inputportssize[i]);
				}
				textstream<<QString("/*! \\def %1_%2_INPUTPORTSSIZE\n").arg(nodetypename).arg(nodeclassname);
				textstream<<QString("\t\\brief Defines input port(s) info: number=%1\n").arg(n);
				textstream<<QString("\t\\details\n");
				for(i=0;i<n;i++)
				{
					textstream<<QString("\t- Input Port #%1: Buffer_Size = %2, Params_Type = %3, Data_Type = %4\n").arg(i).arg(inputportssize.at(i)).arg(inputportsparamstypes.at(i)).arg(inputportsdatatypes.at(i));
				}
				textstream<<QString("*/\n");
				textstream<<QString("#define %1_%2_INPUTPORTSSIZE %3\n\n").arg(nodetypename).arg(nodeclassname).arg(tmpsize);
			}
			else
			{
				textstream<<QString("//There is no input data headers.\n\n");

				textstream<<QString("/*! \\def %1_%2_INPUTPORTSSIZE\n").arg(nodetypename).arg(nodeclassname);
				textstream<<QString("\t\\brief Defines input port(s) info: number=0\n");
				textstream<<QString("*/\n");
				textstream<<QString("#define %1_%2_INPUTPORTSSIZE QList<int>()\n\n").arg(nodetypename).arg(nodeclassname);
			}

			if(paramsheader.size()>0)
			{
				if(ui.InheritParams->isChecked())
				{
					textstream<<QString("//The Params is defined as below\n");
					textstream<<QString("#include%1\n\n").arg(paramsheader);
					writeClass(textstream,"Params", paramstype);
				}
				else
				{
					textstream<<QString("//The Params %1 is defined in the header below\n").arg(paramstype);
					textstream<<QString("#include%1\n\n").arg(paramsheader);
				}
			}
			else
			{
				textstream<<QString("//The Params is defined as below\n");
				writeClass(textstream,"Params");
			}

			if(outputportsnumber>0)
			{
				if(outputdataheader.size()>0)
				{
					if(ui.InheritOutputData->isChecked())
					{
						textstream<<QString("//The Output Data is defined as below\n");
						textstream<<QString("#include%1\n\n").arg(outputdataheader);
						writeClass(textstream,"Data", outputdatatype);
					}
					else
					{
						textstream<<QString("//The Output Data %1 is defined in the header below\n").arg(outputdatatype);
						textstream<<QString("#include%1\n\n").arg(outputdataheader);
					}
				}
				else
				{
					textstream<<QString("//The Output Data is defined as below\n");
					writeClass(textstream,"Data");
				}
			}
			else
			{
				textstream<<QString("//There is no Output Data.\n\n");
			}
			textstream<<QString("/*! \\def %1_%2_OUTPUTPORTSNUMBER\n").arg(nodetypename).arg(nodeclassname);
			textstream<<QString("\t\\brief Defines output port(s) info: number = %1.\n").arg(outputportsnumber);
			textstream<<QString("*/\n");
			textstream<<QString("#define %1_%2_OUTPUTPORTSNUMBER %3\n\n").arg(nodetypename).arg(nodeclassname).arg(outputportsnumber);

			writeTail(textstream);

			file.close();
		}

		file.setFileName(vars_h);
		if(file.open(QIODevice::WriteOnly | QIODevice::Text))
		{
			textstream.setDevice(&file);

			textstream<<QString("//You need to modify this file.\n\n");

			writeHead(textstream,QString("Vars"));

			if(varsheader.size()>0)
			{
				if(ui.InheritVars->isChecked())
				{
					textstream<<QString("//The Vars is defined as below\n");
					textstream<<QString("#include%1\n\n").arg(varsheader);
					writeClass(textstream,"Vars",varstype);
				}
				else
				{
					textstream<<QString("//The Vars %1 is defined in the header below\n").arg(varstype);
					textstream<<QString("#include%1\n\n").arg(varsheader);
				}
			}
			else
			{
				textstream<<QString("//The Vars is defined as below\n");
				writeClass(textstream,"Vars");
			}

			writeTail(textstream);

			file.close();
		}

		file.setFileName(privatefunc_h);
		if(file.open(QIODevice::WriteOnly | QIODevice::Text))
		{
			textstream.setDevice(&file);

			textstream<<QString("//You need not to modify this file.\n\n");

			writeHead(textstream,QString("PrivFunc"));

            textstream<<QString("#include \"../%3/%1_%2_ParamsData.h\"\n").arg(nodetypename).arg(nodeclassname).arg(EDITFOLDER);
            textstream<<QString("#include \"../%3/%1_%2_Vars.h\"\n\n").arg(nodetypename).arg(nodeclassname).arg(EDITFOLDER);

			textstream<<QString("/*! \\def NODECONFIG\n");
			textstream<<QString("\t\\brief Forcefully defines the NodeType_NodeClass.\n");
			textstream<<QString("*/\n");
			textstream<<QString("#ifdef NODECONFIG\n");
			textstream<<QString("#undef NODECONFIG\n");
			textstream<<QString("#endif\n");
			textstream<<QString("#define NODECONFIG %1_%2\n\n").arg(nodetypename).arg(nodeclassname);

			writePrivFuncHeader(textstream);

			writeTail(textstream);

			file.close();
		}

		file.setFileName(privatefunc_cpp);
		if(file.open(QIODevice::WriteOnly | QIODevice::Text))
		{
			textstream.setDevice(&file);

			textstream<<QString("//You need to program this file.\n\n");

            textstream<<QString("#include \"../%3/%1_%2_PrivFunc.h\"\n\n").arg(nodetypename).arg(nodeclassname).arg(NOEDITFOLDER);

        #ifdef Q_OS_LINUX
            textstream<<QString("//*******************Please add static libraries in .pro file*******************\n");
            textstream<<QString("//e.g. unix:LIBS += ... or win32:LIBS += ...\n\n");
        #elif defined(Q_OS_WIN)
			textstream<<QString("//*******************Please add static libraries below*******************\n\n");			
            
			textstream<<QString("#pragma comment(lib,\"Kernel.lib\")\n\n");

			textstream<<QString("#ifdef QT_DEBUG\n\n");

			textstream<<QString("#else\n\n");
            
			textstream<<QString("#endif\n\n");
        #endif
			writePrivFuncCpp(textstream);

			file.close();
		}

		file.setFileName(privatecorefunc_h);
		if(file.open(QIODevice::WriteOnly | QIODevice::Text))
		{
			textstream.setDevice(&file);

			textstream<<QString("//You need not to modify this file.\n\n");

			textstream<<QString("/*! \\defgroup %1_%2 %1_%2\n").arg(nodetypename).arg(nodeclassname);
			textstream<<QString("\t\\ingroup %1_Library %2_NodeClass\n").arg(nodetypename).arg(nodeclassname);
			textstream<<QString("\t\\brief %1_%2 defines the %2 in %1.\n").arg(nodetypename).arg(nodeclassname);
			textstream<<QString("*/\n\n");

			writeHead(textstream,QString("PrivCoreFunc"));

            textstream<<QString("#include \"../%3/%1_%2_ParamsData.h\"\n").arg(nodetypename).arg(nodeclassname).arg(EDITFOLDER);
            textstream<<QString("#include \"../%3/%1_%2_Vars.h\"\n\n").arg(nodetypename).arg(nodeclassname).arg(EDITFOLDER);

			textstream<<QString("/*! \\def NODECONFIG\n");
			textstream<<QString("\t\\brief Forcefully defines the NodeType_NodeClass.\n");
			textstream<<QString("*/\n");
			textstream<<QString("#ifdef NODECONFIG\n");
			textstream<<QString("#undef NODECONFIG\n");
			textstream<<QString("#endif\n");
			textstream<<QString("#define NODECONFIG %1_%2\n\n").arg(nodetypename).arg(nodeclassname);

			writePrivCoreFuncHeader(textstream);

			writeTail(textstream);

			file.close();
		}

		file.setFileName(privatecorefunc_cpp);
		if(file.open(QIODevice::WriteOnly | QIODevice::Text))
		{
			textstream.setDevice(&file);

			textstream<<QString("//Generally you need not to program this file.\n\n");

			textstream<<QString("#include \"%1_%2_PrivCoreFunc.h\"\n\n").arg(nodetypename).arg(nodeclassname);

			writePrivCoreFuncCpp(textstream);

			file.close();
		}
	}

	if(exfuncs.size()>0)
	{
		file.setFileName(privateexfunc_h);
		if(file.open(QIODevice::WriteOnly | QIODevice::Text))
		{
			textstream.setDevice(&file);

			textstream<<QString("//You need not to modify this file.\n\n");

			writeHead(textstream,QString("PrivExFunc"));

            textstream<<QString("#include \"../%3/%1_%2_ParamsData.h\"\n").arg(nodetypename).arg(nodeclassname).arg(EDITFOLDER);
            textstream<<QString("#include \"../%3/%1_%2_Vars.h\"\n\n").arg(nodetypename).arg(nodeclassname).arg(EDITFOLDER);

			textstream<<QString("/*! \\def NODECONFIG\n");
			textstream<<QString("\t\\brief Forcefully defines the NodeType_NodeClass.\n");
			textstream<<QString("*/\n");
			textstream<<QString("#ifdef NODECONFIG\n");
			textstream<<QString("#undef NODECONFIG\n");
			textstream<<QString("#endif\n");
			textstream<<QString("#define NODECONFIG %1_%2\n\n").arg(nodetypename).arg(nodeclassname);

			writePrivExFuncHeader(textstream);

			writeTail(textstream);

			file.close();
		}

		file.setFileName(privateexfunc_cpp);
		if(file.open(QIODevice::WriteOnly | QIODevice::Text))
		{
			textstream.setDevice(&file);

			textstream<<QString("//You need to program this file.\n\n");

            textstream<<QString("#include \"../%3/%1_%2_PrivExFunc.h\"\n\n").arg(nodetypename).arg(nodeclassname).arg(NOEDITFOLDER);

			writePrivExFuncCpp(textstream);

			file.close();
		}
	}
}

void ScanInterfaceFunction::writeHead(QTextStream & textstream, QString suffix)
{
	textstream<<QString("#ifndef %1_%2_%3_H\n").arg(nodetypename.toUpper()).arg(nodeclassname.toUpper()).arg(suffix.toUpper());
	textstream<<QString("#define %1_%2_%3_H\n\n").arg(nodetypename.toUpper()).arg(nodeclassname.toUpper()).arg(suffix.toUpper());

	textstream<<QString("#include<RobotSDK_Global.h>\n\n");

	textstream<<QString("/*! \\defgroup %1_%2_%3 %1_%2_%3\n").arg(nodetypename).arg(nodeclassname).arg(suffix);
	textstream<<QString("\t\\ingroup %1_%2\n").arg(nodetypename).arg(nodeclassname);
	textstream<<QString("\t\\brief %1_%2_%3 defines the %3 in %1_%2.\n").arg(nodetypename).arg(nodeclassname).arg(suffix);
	textstream<<QString("*/\n\n");

	textstream<<QString("/*! \\addtogroup %1_%2_%3\n").arg(nodetypename).arg(nodeclassname).arg(suffix);
	textstream<<QString("\t@{\n");
	textstream<<QString("*/\n\n");

	textstream<<QString("/*! \\file %1_%2_%3.h\n").arg(nodetypename).arg(nodeclassname).arg(suffix);
	textstream<<QString("\t Defines the %3 of %1_%2\n").arg(nodetypename).arg(nodeclassname).arg(suffix);
	textstream<<QString("*/\n\n");

	textstream<<QString("//*******************Please add other headers below*******************\n\n\n");
}

void ScanInterfaceFunction::writeTail(QTextStream & textstream)
{
	textstream<<QString("/*! @}*/ \n\n");
	textstream<<QString("#endif");
}

void ScanInterfaceFunction::writeClass(QTextStream & textstream, QString suffix, QString inherit)
{
	if(inherit.size()>0)
	{
		textstream<<QString("/*! \\class %1_%2_%3 : public %4 \n").arg(nodetypename).arg(nodeclassname).arg(suffix).arg(inherit);
	}
	else
	{
		textstream<<QString("/*! \\class %1_%2_%3 \n").arg(nodetypename).arg(nodeclassname).arg(suffix);
	}
	textstream<<QString("\t\\brief The %3 of %1_%2.\n").arg(nodetypename).arg(nodeclassname).arg(suffix);
	textstream<<QString("\t\\details **Please add details below**\n\n");
	textstream<<QString("*/\n");
	if(inherit.size()>0)
	{
		textstream<<QString("class ROBOTSDK_OUTPUT %1_%2_%3 : public %4 \n").arg(nodetypename).arg(nodeclassname).arg(suffix).arg(inherit);
	}
	else
	{
		textstream<<QString("class ROBOTSDK_OUTPUT %1_%2_%3 \n").arg(nodetypename).arg(nodeclassname).arg(suffix);
	}	
	textstream<<QString("{\n");
	textstream<<QString("public:\n");
	textstream<<QString("\t/*! \\fn %1_%2_%3()\n").arg(nodetypename).arg(nodeclassname).arg(suffix);
	textstream<<QString("\t\t\\brief The constructor of %1_%2_%3. [required]\n").arg(nodetypename).arg(nodeclassname).arg(suffix);
	textstream<<QString("\t\t\\details ****Please add details below****\n\n");
	textstream<<QString("\t*/\n");
	textstream<<QString("\t%1_%2_%3() \n").arg(nodetypename).arg(nodeclassname).arg(suffix);
	textstream<<QString("\t{\n");
	textstream<<QString("\t\t\n");
	textstream<<QString("\t}\n");
	textstream<<QString("\t/*! \\fn ~%1_%2_%3()\n").arg(nodetypename).arg(nodeclassname).arg(suffix);
	textstream<<QString("\t\t\\brief The destructor of %1_%2_%3. [required]\n").arg(nodetypename).arg(nodeclassname).arg(suffix);
	textstream<<QString("\t\t\\details *****Please add details below*****\n\n");
	textstream<<QString("\t*/\n");
	textstream<<QString("\t~%1_%2_%3() \n").arg(nodetypename).arg(nodeclassname).arg(suffix);
	textstream<<QString("\t{\n");
	textstream<<QString("\t\t\n");
	textstream<<QString("\t}\n");
	textstream<<QString("public:\n");
	textstream<<QString("\t//*******************Please add variables below*******************\n\n");
	textstream<<QString("};\n\n");
}

QString ScanInterfaceFunction::getText(QDomElement & tmproot, QStringList tags)
{
	QDomElement curelem=tmproot;
	int i,n=tags.size();
	for(i=0;i<n&&!curelem.isNull();i++)
	{
		curelem=curelem.firstChildElement(tags.at(i));
	}
	if(!curelem.isNull())
	{
		return curelem.text();
	}
	else
	{
		return QString();
	}
}

QStringList ScanInterfaceFunction::getTexts(QDomElement & tmproot, QStringList tags)
{
	QDomElement curelem=tmproot;
	int i,n=tags.size();
	for(i=0;i<n&&!curelem.isNull();i++)
	{
		curelem=curelem.firstChildElement(tags.at(i));
	}
	QStringList result;
	while(!curelem.isNull()&&n>0)
	{
		result<<curelem.text();
		curelem=curelem.nextSiblingElement(tags.back());
	}
	return result;
}

void ScanInterfaceFunction::writePrivFuncHeader(QTextStream & textstream)
{
	int i,n=interfacefunctions.size();
	for(i=0;i<n;i++)
	{
		if(getText(ruleroot,QStringList()<<"FuncRule"<<interfacefunctions[i].functionname<<"FuncType").toInt()>=2)
		{
			QString functioncomment=QString("%1 %3_%4_%2(").arg(interfacefunctions[i].returnvalue).arg(interfacefunctions[i].functionname).arg(nodetypename).arg(nodeclassname);
			QString functiondeclare=QString("%1 DECOFUNC(%2)(").arg(interfacefunctions[i].returnvalue).arg(interfacefunctions[i].functionname);
			int j,m=interfacefunctions[i].parameternames.size();
			for(j=0;j<m;j++)
			{
				functioncomment=functioncomment+QString("%1 %2, ").arg(interfacefunctions[i].parametertypes[j]).arg(interfacefunctions[i].parameternames[j]);
				functiondeclare=functiondeclare+QString("%1 %2, ").arg(interfacefunctions[i].parametertypes[j]).arg(interfacefunctions[i].parameternames[j]);
			}
			if(m>0)
			{
				int tmpsize=functioncomment.size();
				functioncomment.truncate(tmpsize-2);
				tmpsize=functiondeclare.size();
				functiondeclare.truncate(tmpsize-2);
			}
			functioncomment=functioncomment+QString(")");
			functiondeclare=functiondeclare+QString(")");
			QString decofunctiondeclare=QString("extern \"C\" ROBOTSDK_OUTPUT ")+functiondeclare;
			textstream<<QString("/*! %1\n").arg(functioncomment);
			m=interfacefunctions[i].comment.size();
			for(j=0;j<m;j++)
			{
				textstream<<QString("\t%1\n").arg(interfacefunctions[i].comment[j]);
			}
			textstream<<QString("*/\n");
			textstream<<decofunctiondeclare<<";\n\n";
		}
	}
}

void ScanInterfaceFunction::writePrivFuncCpp(QTextStream & textstream)
{
	int i,n=interfacefunctions.size();
	for(i=0;i<n;i++)
	{
		if(getText(ruleroot,QStringList()<<"FuncRule"<<interfacefunctions[i].functionname<<"FuncType").toInt()>=2)
		{
			QString functiondeclare=QString("%1 DECOFUNC(%2)(").arg(interfacefunctions[i].returnvalue).arg(interfacefunctions[i].functionname);
			int j,m=interfacefunctions[i].parameternames.size();
			bool flag=0;
			for(j=0;j<m;j++)
			{
				if(interfacefunctions[i].parameternames[j]=="inputData"||interfacefunctions[i].parameternames[j]=="drainData")
				{
					flag=1;
				}
				functiondeclare=functiondeclare+QString("%1 %2, ").arg(interfacefunctions[i].parametertypes[j]).arg(interfacefunctions[i].parameternames[j]);
			}
			if(m>0)
			{
				int tmpsize=functiondeclare.size();
				functiondeclare.truncate(tmpsize-2);
			}
			functiondeclare=functiondeclare+QString(")");
			m=inputportssize.size();
			for(j=0;j<m&&flag;j++)
			{
				textstream<<QString("//Input Port #%1: Buffer_Size = %2, Params_Type = %3, Data_Type = %4\n").arg(j).arg(inputportssize.at(j)).arg(inputportsparamstypes.at(j)).arg(inputportsdatatypes.at(j));
			}
			textstream<<functiondeclare<<"\n";
			textstream<<"{\n";
			m=interfacefunctions[i].parameternames.size();
			for(j=0;j<m;j++)
			{
				QString tmpparams=getText(ruleroot,QStringList()<<"FuncRule"<<interfacefunctions[i].functionname<<"Parameters"<<interfacefunctions[i].parameternames[j]);
				if(tmpparams!=QString(""))
				{
					replaceText(tmpparams);
					textstream<<QString("\t%1\n").arg(tmpparams);
				}
			}
			QStringList tmpfuncpreset=getTexts(ruleroot,QStringList()<<"FuncRule"<<interfacefunctions[i].functionname<<"FuncPreset");
			m=tmpfuncpreset.size();
			for(j=0;j<m;j++)
			{
				QString tmpqstr=tmpfuncpreset.at(j);
				replaceText(tmpqstr);
				textstream<<QString("\t%1\n").arg(tmpqstr);
			}
			textstream<<"}\n\n";
		}
	}
}

void ScanInterfaceFunction::writePrivCoreFuncHeader(QTextStream & textstream)
{
	int i,n=interfacefunctions.size();
	for(i=0;i<n;i++)
	{
		if(getText(ruleroot,QStringList()<<"FuncRule"<<interfacefunctions[i].functionname<<"FuncType").toInt()==1)
		{
			QString functioncomment=QString("%1 %3_%4_%2(").arg(interfacefunctions[i].returnvalue).arg(interfacefunctions[i].functionname).arg(nodetypename).arg(nodeclassname);
			QString functiondeclare=QString("%1 DECOFUNC(%2)(").arg(interfacefunctions[i].returnvalue).arg(interfacefunctions[i].functionname);
			int j,m=interfacefunctions[i].parameternames.size();
			for(j=0;j<m;j++)
			{
				functioncomment=functioncomment+QString("%1 %2, ").arg(interfacefunctions[i].parametertypes[j]).arg(interfacefunctions[i].parameternames[j]);
				functiondeclare=functiondeclare+QString("%1 %2, ").arg(interfacefunctions[i].parametertypes[j]).arg(interfacefunctions[i].parameternames[j]);
			}
			if(m>0)
			{
				int tmpsize=functioncomment.size();
				functioncomment.truncate(tmpsize-2);
				tmpsize=functiondeclare.size();
				functiondeclare.truncate(tmpsize-2);
			}
			functioncomment=functioncomment+QString(")");
			functiondeclare=functiondeclare+QString(")");
			QString decofunctiondeclare=QString("extern \"C\" ROBOTSDK_OUTPUT ")+functiondeclare;
			textstream<<QString("/*! %1\n").arg(functioncomment);
			m=interfacefunctions[i].comment.size();
			for(j=0;j<m;j++)
			{
				textstream<<QString("\t%1\n").arg(interfacefunctions[i].comment[j]);
			}
			textstream<<QString("*/\n");
			textstream<<decofunctiondeclare<<";\n\n";
		}
	}
}

void ScanInterfaceFunction::writePrivCoreFuncCpp(QTextStream & textstream)
{
	int i,n=interfacefunctions.size();
	for(i=0;i<n;i++)
	{
		if(getText(ruleroot,QStringList()<<"FuncRule"<<interfacefunctions[i].functionname<<"FuncType").toInt()==1)
		{
			QString functiondeclare=QString("%1 DECOFUNC(%2)(").arg(interfacefunctions[i].returnvalue).arg(interfacefunctions[i].functionname);
			int j,m=interfacefunctions[i].parameternames.size();
			bool flag=0;
			for(j=0;j<m;j++)
			{
				if(interfacefunctions[i].parameternames[j]=="inputData"||interfacefunctions[i].parameternames[j]=="drainData")
				{
					flag=1;
				}
				functiondeclare=functiondeclare+QString("%1 %2, ").arg(interfacefunctions[i].parametertypes[j]).arg(interfacefunctions[i].parameternames[j]);
			}
			if(m>0)
			{
				int tmpsize=functiondeclare.size();
				functiondeclare.truncate(tmpsize-2);
			}
			functiondeclare=functiondeclare+QString(")");
			m=inputportssize.size();
			for(j=0;j<m&&flag;j++)
			{
				textstream<<QString("//Input Port #%1: Buffer_Size = %2, Params_Type = %3, Data_Type = %4\n").arg(j).arg(inputportssize.at(j)).arg(inputportsparamstypes.at(j)).arg(inputportsdatatypes.at(j));
			}
			textstream<<functiondeclare<<"\n";
			textstream<<"{\n";
			m=interfacefunctions[i].parameternames.size();
			for(j=0;j<m;j++)
			{
				QString tmpparams=getText(ruleroot,QStringList()<<"FuncRule"<<interfacefunctions[i].functionname<<"Parameters"<<interfacefunctions[i].parameternames[j]);
				if(tmpparams!=QString(""))
				{
					replaceText(tmpparams);
					textstream<<QString("\t%1\n").arg(tmpparams);
				}
			}
			QStringList tmpfuncpreset=getTexts(ruleroot,QStringList()<<"FuncRule"<<interfacefunctions[i].functionname<<"FuncPreset");
			m=tmpfuncpreset.size();
			for(j=0;j<m;j++)
			{
				QString tmpqstr=tmpfuncpreset.at(j);
				replaceText(tmpqstr);
				textstream<<QString("\t%1\n").arg(tmpqstr);
			}
			textstream<<"}\n\n";
		}
	}
}

void ScanInterfaceFunction::writePrivExFuncHeader(QTextStream & textstream)
{
	int k,l=exfuncs.size();
	for(k=0;k<l;k++)
	{
		int i,n=interfacefunctions.size();
		for(i=0;i<n;i++)
		{
			if(getText(ruleroot,QStringList()<<"FuncRule"<<interfacefunctions[i].functionname<<"FuncType").toInt()==3)
			{
				QString functioncomment=QString("%1 %3_%4_%2_%5(").arg(interfacefunctions[i].returnvalue).arg(interfacefunctions[i].functionname).arg(nodetypename).arg(nodeclassname).arg(exfuncs.at(k));
				QString functiondeclare=QString("%1 DECOFUNC(%2_%3)(").arg(interfacefunctions[i].returnvalue).arg(interfacefunctions[i].functionname).arg(exfuncs.at(k));
				int j,m=interfacefunctions[i].parameternames.size();
				for(j=0;j<m;j++)
				{
					functioncomment=functioncomment+QString("%1 %2, ").arg(interfacefunctions[i].parametertypes[j]).arg(interfacefunctions[i].parameternames[j]);
					functiondeclare=functiondeclare+QString("%1 %2, ").arg(interfacefunctions[i].parametertypes[j]).arg(interfacefunctions[i].parameternames[j]);
				}
				if(m>0)
				{
					int tmpsize=functioncomment.size();
					functioncomment.truncate(tmpsize-2);
					tmpsize=functiondeclare.size();
					functiondeclare.truncate(tmpsize-2);
				}
				functioncomment=functioncomment+QString(")");
				functiondeclare=functiondeclare+QString(")");
				QString decofunctiondeclare=QString("extern \"C\" ROBOTSDK_OUTPUT ")+functiondeclare;
				textstream<<QString("/*! %1\n").arg(functioncomment);
				m=interfacefunctions[i].comment.size();
				for(j=0;j<m;j++)
				{
					textstream<<QString("\t%1\n").arg(interfacefunctions[i].comment[j]);
				}
				textstream<<QString("*/\n");
				textstream<<decofunctiondeclare<<";\n\n";
			}
		}
	}
}

void ScanInterfaceFunction::writePrivExFuncCpp(QTextStream & textstream)
{
	int k,l=exfuncs.size();
	for(k=0;k<l;k++)
	{
		int i,n=interfacefunctions.size();
		for(i=0;i<n;i++)
		{
			if(getText(ruleroot,QStringList()<<"FuncRule"<<interfacefunctions[i].functionname<<"FuncType").toInt()==3)
			{
				QString functiondeclare=QString("%1 DECOFUNC(%2_%3)(").arg(interfacefunctions[i].returnvalue).arg(interfacefunctions[i].functionname).arg(exfuncs.at(k));
				int j,m=interfacefunctions[i].parameternames.size();
				bool flag=0;
				for(j=0;j<m;j++)
				{
					if(interfacefunctions[i].parameternames[j]=="inputData"||interfacefunctions[i].parameternames[j]=="drainData")
					{
						flag=1;
					}
					functiondeclare=functiondeclare+QString("%1 %2, ").arg(interfacefunctions[i].parametertypes[j]).arg(interfacefunctions[i].parameternames[j]);
				}
				if(m>0)
				{
					int tmpsize=functiondeclare.size();
					functiondeclare.truncate(tmpsize-2);
				}
				functiondeclare=functiondeclare+QString(")");
				m=inputportssize.size();
				for(j=0;j<m&&flag;j++)
				{
					textstream<<QString("//Input Port #%1: Buffer_Size = %2, Params_Type = %3, Data_Type = %4\n").arg(j).arg(inputportssize.at(j)).arg(inputportsparamstypes.at(j)).arg(inputportsdatatypes.at(j));
				}
				textstream<<functiondeclare<<"\n";
				textstream<<"{\n";
				m=interfacefunctions[i].parameternames.size();
				for(j=0;j<m;j++)
				{
					QString tmpparams=getText(ruleroot,QStringList()<<"FuncRule"<<interfacefunctions[i].functionname<<"Parameters"<<interfacefunctions[i].parameternames[j]);
					if(tmpparams!=QString(""))
					{
						replaceText(tmpparams);
						textstream<<QString("\t%1\n").arg(tmpparams);
					}
				}
				QStringList tmpfuncpreset=getTexts(ruleroot,QStringList()<<"FuncRule"<<interfacefunctions[i].functionname<<"FuncPreset");
				m=tmpfuncpreset.size();
				for(j=0;j<m;j++)
				{
					QString tmpqstr=tmpfuncpreset.at(j);
					replaceText(tmpqstr);
					textstream<<QString("\t%1\n").arg(tmpqstr);
				}
				textstream<<"}\n\n";
			}
		}
	}
}

void ScanInterfaceFunction::replaceText(QString & text)
{
	text.replace("$(NodeType)",nodetypename);
	text.replace("$(NodeClass)",nodeclassname);
	
	if(paramstype.size()>0&&!(ui.InheritParams->isChecked()))
	{
		text.replace("$(ParamsType)",paramstype);
	}
	else
	{
		text.replace("$(ParamsType)",QString("%1_%2_Params").arg(nodetypename).arg(nodeclassname));
	}

	if(outputdatatype.size()>0&&!(ui.InheritOutputData->isChecked()))
	{
		text.replace("$(OutputDataType)",outputdatatype);
	}
	else
	{
		text.replace("$(OutputDataType)",QString("%1_%2_Data").arg(nodetypename).arg(nodeclassname));
	}

	if(varstype.size()>0&&!(ui.InheritVars->isChecked()))
	{
		text.replace("$(VarsType)",varstype);
	}
	else
	{
		text.replace("$(VarsType)",QString("%1_%2_Vars").arg(nodetypename).arg(nodeclassname));
	}

	int i,n=inputportsparamstypes.size();
	if(text.contains("$(Index)")&&n>0)
	{
		QString tmpqstr;
		for(i=0;i<n;i++)
		{
			QString tmptext=text;
			tmptext.replace("$(InputParamsType)",inputportsparamstypes.at(i));
			tmptext.replace("$(InputDataType)",inputportsdatatypes.at(i));
			tmptext.replace("$(Index)",QString("%1").arg(i));
			if(i<n-1)
			{
				tmpqstr=tmpqstr+tmptext+QString("\n\t");
			}
			else
			{
				tmpqstr=tmpqstr+tmptext;
			}
		}
		text=tmpqstr;
	}
	else if(n==1)
	{
		text.replace("$(InputParamsType)",inputportsparamstypes.front());
		text.replace("$(InputDataType)",inputportsdatatypes.front());
	}

	text.replace("$(InputPortsSize)",QString("%1_%2_INPUTPORTSSIZE").arg(nodetypename).arg(nodeclassname));

	text.replace("$(OutputPortsNumber)",QString("%1_%2_OUTPUTPORTSNUMBER").arg(nodetypename).arg(nodeclassname));
}

void ScanInterfaceFunction::configQtProject()
{
	QStringList headerlist;
	QStringList cpplist;
	QStringList installheaderslist;
	QString editpath = nodeclassname;
	QString noeditpath = nodeclassname;
	editpath = QString("%1/%2/%3").arg(editpath.replace("_", "/")).arg(nodetypename).arg(EDITFOLDER);
	noeditpath = QString("%1/%2/%3").arg(noeditpath.replace("_", "/")).arg(nodetypename).arg(NOEDITFOLDER);
	if (!ui.OnlyExFunc->isChecked())
	{
		headerlist << QString("%3/%1_%2_ParamsData.h").arg(nodetypename).arg(nodeclassname).arg(editpath)
			<< QString("%3/%1_%2_Vars.h").arg(nodetypename).arg(nodeclassname).arg(editpath)
			<< QString("%3/%1_%2_PrivFunc.h").arg(nodetypename).arg(nodeclassname).arg(noeditpath)
			<< QString("%3/%1_%2_PrivCoreFunc.h").arg(nodetypename).arg(nodeclassname).arg(noeditpath);
		cpplist << QString("%3/%1_%2_PrivFunc.cpp").arg(nodetypename).arg(nodeclassname).arg(editpath)
			<< QString("%3/%1_%2_PrivCoreFunc.cpp").arg(nodetypename).arg(nodeclassname).arg(noeditpath);
		installheaderslist << QString("%3/%1_%2_ParamsData.h").arg(nodetypename).arg(nodeclassname).arg(editpath)
			<< QString("%3/%1_%2_Vars.h").arg(nodetypename).arg(nodeclassname).arg(editpath);
	}
	if (exfuncs.size() > 0)
	{
		headerlist << QString("%3/%1_%2_PrivExFunc.h").arg(nodetypename).arg(nodeclassname).arg(noeditpath);
		cpplist << QString("%3/%1_%2_PrivExFunc.cpp").arg(nodetypename).arg(nodeclassname).arg(editpath);
	}

	QFile file;
	QTextStream stream;
	QString textcontent;
	QString filename = outputvcxproj;
	QFileInfo fileinfo(filename);

	file.setFileName(filename);
	if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
	{
		textcontent.clear();
	}
	else
	{
		textcontent = file.readAll();
		file.close();
	}
	if (textcontent.contains("TEMPLATE = subdirs"))
	{
		QMessageBox::information(this, "Error", "This is a subdirs project");
		return;
	}
	if (!textcontent.contains(QString("PROJNAME = %1").arg(fileinfo.baseName())))
	{
		textcontent = textcontent + QString("\nPROJNAME = %1").arg(fileinfo.baseName());
	}
	if (!textcontent.contains("INSTTYPE = MOD"))
	{
		textcontent = textcontent + QString("\nINSTTYPE = MOD");
	}
	if (textcontent.contains("INSTTYPE = APP"))
	{
		textcontent.remove(QString("\nINSTTYPE = APP"));
	}
	if (textcontent.contains("INSTTYPE = SDK"))
	{
		textcontent.remove(QString("\nINSTTYPE = SDK"));
	}
	if (!textcontent.contains("include(RobotSDK_Main.pri)"))
	{
		textcontent = textcontent + QString("\ninclude(RobotSDK_Main.pri)");
	}
	else
	{
		textcontent.remove(QString("\ninclude(RobotSDK_Main.pri)"));
		textcontent = textcontent + QString("\ninclude(RobotSDK_Main.pri)");
	}
	if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
	{
		return;
	}
	stream.setDevice(&file);
	stream << textcontent;
	file.close();
   

    QString prifile=QString("%1/%2.pri").arg(fileinfo.absolutePath()).arg(fileinfo.baseName());
    file.setFileName(prifile);
    QString sources;
    QString headers;
    if(file.open(QIODevice::ReadOnly|QIODevice::Text))
    {
        while(!file.atEnd())
        {
            QString tmpline=file.readLine();
            while(!file.atEnd()&&tmpline.endsWith("\\\n"))
            {
                tmpline+=file.readLine();
            }
            if(tmpline.endsWith("\n"))
            {
                tmpline.truncate(tmpline.size()-1);
            }
            if(tmpline.startsWith("SOURCES +="))
            {
                sources=tmpline;
                int i,n=cpplist.size();
                for(i=0;i<n;i++)
                {
                    if(!sources.contains(QString("\\\n\t./")+cpplist.at(i)+QString("\t")))
                    {
                        sources=sources+QString("\\\n\t./")+cpplist.at(i)+QString("\t");
                    }
                }
            }
            else if(tmpline.startsWith("HEADERS +="))
            {
                headers=tmpline;
                int i,n=headerlist.size();
                for(i=0;i<n;i++)
                {
                    if(!headers.contains(QString("\\\n\t./")+headerlist.at(i)+QString("\t")))
                    {
                        headers=headers+QString("\\\n\t./")+headerlist.at(i)+QString("\t");
                    }
                }
            }
        }
        file.close();
    }
    if(!file.open(QIODevice::WriteOnly|QIODevice::Text))
    {
        return;
    }
    stream.setDevice(&file);
    if(sources.size()>0)
    {
        stream<<sources<<"\n\n";
    }
    else
    {
        sources="SOURCES += ";
        int i,n=cpplist.size();
        for(i=0;i<n;i++)
        {
            if(!sources.contains(QString("\\\n\t./")+cpplist.at(i)+QString("\t")))
            {
                sources=sources+QString("\\\n\t./")+cpplist.at(i)+QString("\t");
            }
        }
        stream<<sources<<"\n\n";
    }
    if(headers.size()>0)
    {
        stream<<headers<<"\n\n";
    }
    else
    {
        headers="HEADERS += ";
        int i,n=headerlist.size();
        for(i=0;i<n;i++)
        {
            if(!headers.contains(QString("\\\n\t./")+headerlist.at(i)+QString("\t")))
            {
                headers=headers+QString("\\\n\t./")+headerlist.at(i)+QString("\t");
            }
        }
        stream<<headers<<"\n\n";
    }
    file.close();

    file.setFileName(QString("%1/RobotSDK_Main.pri").arg(ROBOTSDKTOOLS));
    if(!file.open(QIODevice::ReadOnly|QIODevice::Text))
    {
        return;
    }
    textcontent=file.readAll();
    file.close();
    file.setFileName(QString("%1/RobotSDK_Main.pri").arg(fileinfo.absolutePath()));
    file.open(QIODevice::WriteOnly|QIODevice::Text);
    stream.setDevice(&file);
    stream<<textcontent;
    file.close();

    file.setFileName(QString("%1/RobotSDK_Install.pri").arg(ROBOTSDKTOOLS));
    if(!file.open(QIODevice::ReadOnly|QIODevice::Text))
    {
        return;
    }
    textcontent=file.readAll();
    file.close();
    file.setFileName(QString("%1/RobotSDK_Install.pri").arg(fileinfo.absolutePath()));
    file.open(QIODevice::WriteOnly|QIODevice::Text);
    stream.setDevice(&file);
    stream<<textcontent;
    file.close();
    return;
}

void ScanInterfaceFunction::configVSProject()
{
	QDomDocument * projectdoc=new QDomDocument();
	QString filename=outputvcxproj;
	QFile file(filename);
	if(file.open(QIODevice::ReadOnly|QIODevice::Text))
	{
		if(!projectdoc->setContent(&file))
		{
			delete projectdoc;
			file.close();
			return;
		}
		file.close();
	}
	else
	{
		return;
	}

	QDomElement projectroot=projectdoc->documentElement();
	QDomElement propertygroup=projectroot.firstChildElement("PropertyGroup");
	while(propertygroup.attribute("Label")!="UserMacros")
	{
		propertygroup=propertygroup.nextSiblingElement("PropertyGroup");
	}
	if(propertygroup.isNull())
	{
		propertygroup=projectroot.appendChild(projectdoc->createElement("PropertyGroup")).toElement();
		propertygroup.setAttribute("Label","UserMacros");
	}
	propertygroup=propertygroup.nextSiblingElement("PropertyGroup");
	QStringList conditions;
#ifdef Q_PROCESSOR_X86_64
	conditions<<"\'$(Configuration)|$(Platform)\'==\'Debug|Win32\'";
	conditions<<"\'$(Configuration)|$(Platform)\'==\'Release|Win32\'";
	conditions<<"\'$(Configuration)|$(Platform)\'==\'Debug|x64\'";
	conditions<<"\'$(Configuration)|$(Platform)\'==\'Release|x64\'";
#elif Q_PROCESSOR_X86
    conditions<<"\'$(Configuration)|$(Platform)\'==\'Debug|Win32\'";
    conditions<<"\'$(Configuration)|$(Platform)\'==\'Release|Win32\'";
#endif
	int j,m=conditions.size();
	for(j=0;j<m;j++)
	{
		if(propertygroup.isNull())
		{
			propertygroup=projectroot.appendChild(projectdoc->createElement("PropertyGroup")).toElement();
		}
		propertygroup.setAttribute("Condition",conditions.at(j));
        setText(projectdoc,propertygroup,"OutDir","$(RobotSDK_SharedLibrary)\\");
        setText(projectdoc,propertygroup,"IncludePath","$(RobotSDK_Kernel)\\include;$(RobotSDK_ModuleDev);$(RobotSDK_Module);$(RobotDep_Include);$(IncludePath)");
        setText(projectdoc,propertygroup,"LibraryPath","$(RobotSDK_Kernel)\\lib\\$(Configuration);$(RobotDep_Lib);$(LibraryPath)");
		propertygroup=propertygroup.nextSiblingElement("PropertyGroup");
	}
	while(!propertygroup.isNull())
	{
		QDomElement tmpelem=propertygroup.nextSiblingElement("PropertyGroup");
		projectroot.removeChild(propertygroup).clear();
		propertygroup=tmpelem;
	}

	QDomElement itemdefine=projectroot.firstChildElement("ItemDefinitionGroup");
	while(!itemdefine.isNull())
	{
		int conditionsid=conditions.indexOf(itemdefine.attribute("Condition"));
		if(conditionsid<0)
		{
			continue;
		}
		QDomElement predefineelem=itemdefine.firstChildElement("ClCompile").firstChildElement("PreprocessorDefinitions");
		QString predefine=predefineelem.text();
        if(!predefine.contains("RobotSDK_ModuleDev"))
		{
            predefine=QString("RobotSDK_ModuleDev;")+predefine;
			QDomNode tmpnode=predefineelem.firstChild();
			while(!tmpnode.isNull()&&!tmpnode.isText())
			{
				tmpnode=tmpnode.nextSibling();
			}
			if(!tmpnode.isNull())
			{
				tmpnode.toText().setData(predefine);
			}
		}
		QDomElement outputfileelem=itemdefine.firstChildElement("Link").firstChildElement("OutputFile");
		QString outputfile=outputfileelem.text();
        outputfile=QString("$(OutDir)\\$(ProjectName)_$(Configuration).dll");
		QDomNode tmpnode=outputfileelem.firstChild();
		while(!tmpnode.isNull()&&!tmpnode.isText())
		{
			tmpnode=tmpnode.nextSibling();
		}
		if(!tmpnode.isNull())
		{
			tmpnode.toText().setData(outputfile);
		}
		itemdefine=itemdefine.nextSiblingElement("ItemDefinitionGroup");
	}

	QStringList headerlist;
	QStringList cpplist;
    QString editpath=nodeclassname;
    QString noeditpath=nodeclassname;
    editpath=QString("%1/%2/%3").arg(editpath.replace("_","/")).arg(nodetypename).arg(EDITFOLDER);
    noeditpath=QString("%1/%2/%3").arg(noeditpath.replace("_","/")).arg(nodetypename).arg(NOEDITFOLDER);
    if(!ui.OnlyExFunc->isChecked())
	{
        headerlist<<QString("%3/%1_%2_ParamsData.h").arg(nodetypename).arg(nodeclassname).arg(editpath)
            <<QString("%3/%1_%2_Vars.h").arg(nodetypename).arg(nodeclassname).arg(editpath)
            <<QString("%3/%1_%2_PrivFunc.h").arg(nodetypename).arg(nodeclassname).arg(noeditpath)
            <<QString("%3/%1_%2_PrivCoreFunc.h").arg(nodetypename).arg(nodeclassname).arg(noeditpath);
        cpplist<<QString("%3/%1_%2_PrivFunc.cpp").arg(nodetypename).arg(nodeclassname).arg(editpath)
            <<QString("%3/%1_%2_PrivCoreFunc.cpp").arg(nodetypename).arg(nodeclassname).arg(noeditpath);
	}
	if(exfuncs.size()>0)
	{
        headerlist<<QString("%3/%1_%2_PrivExFunc.h").arg(nodetypename).arg(nodeclassname).arg(noeditpath);
        cpplist<<QString("%3/%1_%2_PrivExFunc.cpp").arg(nodetypename).arg(nodeclassname).arg(editpath);
	}

	QDomElement itemgroup=projectroot.firstChildElement("ItemGroup");
	bool hasinclude=0;
	bool hascompile=0;
	while(!itemgroup.isNull())
	{
		QDomElement tmpelem=itemgroup.firstChildElement();
		QString tmptag=tmpelem.nodeName();
		if(tmptag=="ClInclude")
		{
			hasinclude=1;
			int i,n=headerlist.size();
			for(i=0;i<n;i++)
			{
				QDomElement headerelem=tmpelem;
				while(!headerelem.isNull())
				{
					if(headerelem.attribute("Include")==headerlist.at(i))
					{
						break;
					}
					headerelem=headerelem.nextSiblingElement("ClInclude");
				}
				if(headerelem.isNull())
				{
					itemgroup.appendChild(projectdoc->createElement("ClInclude")).toElement().setAttribute("Include",headerlist.at(i));
				}
			}
		}
		else if(tmptag=="ClCompile")
		{
			hascompile=1;
			int i,n=cpplist.size();
			for(i=0;i<n;i++)
			{
				QDomElement cppelem=tmpelem;
				while(!cppelem.isNull())
				{
					if(cppelem.attribute("Include")==cpplist.at(i))
					{
						break;
					}
					cppelem=cppelem.nextSiblingElement("ClCompile");
				}
				if(cppelem.isNull())
				{
					itemgroup.appendChild(projectdoc->createElement("ClCompile")).toElement().setAttribute("Include",cpplist.at(i));
				}
			}
		}
		itemgroup=itemgroup.nextSiblingElement("ItemGroup");
	}

	if(!hasinclude)
	{
		itemgroup=projectroot.appendChild(doc->createElement("ItemGroup")).toElement();
		int i,n=headerlist.size();
		for(i=0;i<n;i++)
		{
			itemgroup.appendChild(projectdoc->createElement("ClInclude")).toElement().setAttribute("Include",headerlist.at(i));
		}
	}

	if(!hascompile)
	{
		itemgroup=projectroot.appendChild(doc->createElement("ItemGroup")).toElement();
		int i,n=cpplist.size();
		for(i=0;i<n;i++)
		{
			itemgroup.appendChild(projectdoc->createElement("ClCompile")).toElement().setAttribute("Include",cpplist.at(i));
		}
	}

	file.open(QIODevice::WriteOnly | QIODevice::Text);
	QTextStream textstream(&file);
	projectdoc->save(textstream,2);
	file.close();
	delete projectdoc;

	QDomDocument * filterdoc=new QDomDocument();
	filename=outputvcxproj+QString(".filters");
	QFile filterfile(filename);
	if(filterfile.open(QIODevice::ReadOnly|QIODevice::Text))
	{
		if(!filterdoc->setContent(&filterfile))
		{
			delete filterdoc;
			filterfile.close();
			return;
		}
		filterfile.close();
	}
	else
	{
		return;
	}

	QDomElement filterroot=filterdoc->documentElement();
	itemgroup=filterroot.firstChildElement("ItemGroup");
	if(itemgroup.isNull())
	{
		itemgroup=filterroot.appendChild(filterdoc->createElement("ItemGroup")).toElement();
	}
	QStringList path=nodeclassname.split("_",QString::SkipEmptyParts);
	QString filters=QString("ShellCode");
	QDomElement filter=itemgroup.firstChildElement("Filter");
	bool flag=1;
	int i,n=path.size();
	for(i=0;i<n;i++)
	{		
		while(!filter.isNull())
		{
			if(filter.attribute("Include")==filters)
			{
				flag=0;
				break;
			}
			filter=filter.nextSiblingElement("Filter");
		}
		if(flag)
		{
			filter=itemgroup.appendChild(filterdoc->createElement("Filter")).toElement();
			filter.setAttribute("Include",filters);
			filter.appendChild(filterdoc->createElement("Extensions")).appendChild(filterdoc->createTextNode("h;cpp"));
		}
		filters=filters+QString("\\")+path.at(i);
		filter=itemgroup.firstChildElement("Filter");
		flag=1;
	}
	while(!filter.isNull())
	{
		if(filter.attribute("Include")==filters)
		{
			flag=0;
			break;
		}
		filter=filter.nextSiblingElement("Filter");
	}
	if(flag)
	{
		filter=itemgroup.appendChild(filterdoc->createElement("Filter")).toElement();
		filter.setAttribute("Include",filters);
		filter.appendChild(filterdoc->createElement("Extensions")).appendChild(filterdoc->createTextNode("h;cpp"));
	}

	filters=filters+QString("\\")+nodetypename;
	filter=itemgroup.firstChildElement("Filter");
	flag=1;
	while(!filter.isNull())
	{
		if(filter.attribute("Include")==filters)
		{
			flag=0;
			break;
		}
		filter=filter.nextSiblingElement("Filter");
	}
	if(flag)
	{
		filter=itemgroup.appendChild(filterdoc->createElement("Filter")).toElement();
		filter.setAttribute("Include",filters);
		filter.appendChild(filterdoc->createElement("Extensions")).appendChild(filterdoc->createTextNode("h;cpp"));
	}

	QString noedit=filters+QString("\\NoEdit");
	filter=itemgroup.firstChildElement("Filter");
	flag=1;
	while(!filter.isNull())
	{
		if(filter.attribute("Include")==noedit)
		{
			flag=0;
			break;
		}
		filter=filter.nextSiblingElement("Filter");
	}
	if(flag)
	{
		filter=itemgroup.appendChild(filterdoc->createElement("Filter")).toElement();
		filter.setAttribute("Include",noedit);
		filter.appendChild(filterdoc->createElement("Extensions")).appendChild(filterdoc->createTextNode("h;cpp"));
	}

	QString edit=filters+QString("\\Edit");
	filter=itemgroup.firstChildElement("Filter");
	flag=1;
	while(!filter.isNull())
	{
		if(filter.attribute("Include")==edit)
		{
			flag=0;
			break;
		}
		filter=filter.nextSiblingElement("Filter");
	}
	if(flag)
	{
		filter=itemgroup.appendChild(filterdoc->createElement("Filter")).toElement();
		filter.setAttribute("Include",edit);
		filter.appendChild(filterdoc->createElement("Extensions")).appendChild(filterdoc->createTextNode("h;cpp"));
	}

	QStringList noeditheaderlist;
	QStringList noeditcpplist;
	QStringList editheaderlist;
	QStringList editcpplist;
	if(!ui.OnlyExFunc->isChecked())
	{
        noeditheaderlist<<QString("%3/%1_%2_PrivFunc.h").arg(nodetypename).arg(nodeclassname).arg(noeditpath)
            <<QString("%3/%1_%2_PrivCoreFunc.h").arg(nodetypename).arg(nodeclassname).arg(noeditpath);
        noeditcpplist<<QString("%3/%1_%2_PrivCoreFunc.cpp").arg(nodetypename).arg(nodeclassname).arg(noeditpath);
        editheaderlist<<QString("%3/%1_%2_ParamsData.h").arg(nodetypename).arg(nodeclassname).arg(editpath)
            <<QString("%3/%1_%2_Vars.h").arg(nodetypename).arg(nodeclassname).arg(editpath);
        editcpplist<<QString("%3/%1_%2_PrivFunc.cpp").arg(nodetypename).arg(nodeclassname).arg(editpath);
	}
	if(exfuncs.size()>0)
	{
        noeditheaderlist<<QString("%3/%1_%2_PrivExFunc.h").arg(nodetypename).arg(nodeclassname).arg(noeditpath);
        editcpplist<<QString("%3/%1_%2_PrivExFunc.cpp").arg(nodetypename).arg(nodeclassname).arg(editpath);
	}

	itemgroup=itemgroup.nextSiblingElement("ItemGroup");
	if(itemgroup.isNull())
	{
		itemgroup=filterroot.appendChild(filterdoc->createElement("ItemGroup")).toElement();
	}
	QDomElement ClCompile=itemgroup.firstChildElement("ClCompile");
	while(!ClCompile.isNull())
	{
		QString filename=ClCompile.attribute("Include");
		noeditcpplist.removeAll(filename);
		editcpplist.removeAll(filename);
		ClCompile=ClCompile.nextSiblingElement("ClCompile");
	}
	n=noeditcpplist.size();
	for(i=0;i<n;i++)
	{
		ClCompile=itemgroup.appendChild(filterdoc->createElement("ClCompile")).toElement();
		ClCompile.setAttribute("Include",noeditcpplist.at(i));
		ClCompile.appendChild(filterdoc->createElement("Filter")).appendChild(filterdoc->createTextNode(noedit));
	}
	n=editcpplist.size();
	for(i=0;i<n;i++)
	{
		ClCompile=itemgroup.appendChild(filterdoc->createElement("ClCompile")).toElement();
		ClCompile.setAttribute("Include",editcpplist.at(i));
		ClCompile.appendChild(filterdoc->createElement("Filter")).appendChild(filterdoc->createTextNode(edit));
	}

	itemgroup=itemgroup.nextSiblingElement("ItemGroup");
	if(itemgroup.isNull())
	{
		itemgroup=filterroot.appendChild(filterdoc->createElement("ItemGroup")).toElement();
	}
	QDomElement ClInclude=itemgroup.firstChildElement("ClInclude");
	while(!ClInclude.isNull())
	{
		QString filename=ClInclude.attribute("Include");
		noeditheaderlist.removeAll(filename);
		editheaderlist.removeAll(filename);
		ClInclude=ClInclude.nextSiblingElement("ClInclude");
	}
	n=noeditheaderlist.size();
	for(i=0;i<n;i++)
	{
		ClInclude=itemgroup.appendChild(filterdoc->createElement("ClInclude")).toElement();
		ClInclude.setAttribute("Include",noeditheaderlist.at(i));
		ClInclude.appendChild(filterdoc->createElement("Filter")).appendChild(filterdoc->createTextNode(noedit));
	}
	n=editheaderlist.size();
	for(i=0;i<n;i++)
	{
		ClInclude=itemgroup.appendChild(filterdoc->createElement("ClInclude")).toElement();
		ClInclude.setAttribute("Include",editheaderlist.at(i));
		ClInclude.appendChild(filterdoc->createElement("Filter")).appendChild(filterdoc->createTextNode(edit));
	}

	filterfile.open(QIODevice::WriteOnly | QIODevice::Text);
	QTextStream filtertextstream(&filterfile);
	filterdoc->save(filtertextstream,2);
	filterfile.close();
	delete filterdoc;
}

void ScanInterfaceFunction::setText(QDomDocument * tmpdoc, QDomElement & tmproot, QString tag, QString text)
{
	QDomElement curelem=tmproot.firstChildElement(tag);
	if(curelem.isNull())
	{
		curelem=tmproot.appendChild(tmpdoc->createElement(tag)).toElement();
	}
	if(curelem.hasChildNodes())
	{
		QDomNode tmpnode=curelem.firstChild();
		while(!tmpnode.isNull()&&!tmpnode.isText())
		{
			tmpnode=tmpnode.nextSibling();
		}
		if(!tmpnode.isNull())
		{
			tmpnode.toText().setData(text);
		}
		else
		{
			curelem.appendChild(tmpdoc->createTextNode(text));
		}
	}
	else
	{
		curelem.appendChild(tmpdoc->createTextNode(text));
	}
}
