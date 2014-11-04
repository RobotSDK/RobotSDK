#include "configproject.h"

ConfigProject::ConfigProject(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);

	connect(ui.Browse,SIGNAL(clicked()),this,SLOT(browseSlot()));
	connect(ui.Config,SIGNAL(clicked()),this,SLOT(configSlot()));

#ifdef Q_OS_LINUX
    ui.VSVersion->setEnabled(0);
#elif defined(Q_OS_WIN)
    vversion<<"v120"<<"v110"<<"v100"<<"v90"<<"v80";
    formatversion<<"12.0"<<"11.00"<<"10.00"<<"9.0"<<"8.0";
    vsversion<<"2013"<<"2012"<<"2010"<<"2009"<<"2008";
    ui.VSVersion->addItems(vversion);
#endif

}

ConfigProject::~ConfigProject()
{

}

void ConfigProject::browseSlot()
{
	QString startdir;
	if(ui.Application->isChecked())
	{
		startdir=QString();
	}
	else if(ui.Library->isChecked())
	{
        startdir=ROBOTSDKMODULEDEV;
	}
	else if (ui.SDK->isChecked())
	{
		startdir = QString();
	}
	if (ui.Recursive->isChecked())
	{
		QString projectsdir = QFileDialog::getExistingDirectory(this, "Project Dir", startdir);
		ui.ProjectsDir->setText(projectsdir);
	}
	else
	{
		QString projectsdir = QFileDialog::getOpenFileName(this, "Project Dir", startdir, QString("Project File (*.vcxproj *.pro)"));		
		ui.ProjectsDir->setText(projectsdir);
	}	
}

void ConfigProject::configSlot()
{
	QString projectsdir=ui.ProjectsDir->text();
	configProjects(projectsdir);
	QMessageBox::information(this,"Finish","Finish Configuration.");
}

void ConfigProject::setText(QDomDocument * tmpdoc, QDomElement & tmproot, QString tag, QString text)
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

void ConfigProject::configProjects(QString projectsdir)
{
	QFileInfo tmpinfo=QFileInfo(projectsdir);
	if (tmpinfo.isDir())
	{
		QDir dir(projectsdir);
		if (!dir.exists())
		{
			return;
		}
		dir.setFilter(QDir::Dirs | QDir::Files | QDir::NoDotAndDotDot);
		dir.setSorting(QDir::DirsFirst);
		QFileInfoList list = dir.entryInfoList();
		int i, n = list.size();
		for (i = 0; i<n; i++)
		{
			QFileInfo fileinfo = list.at(i);
			QString filename = fileinfo.absoluteFilePath();
			if (fileinfo.isDir() && ui.Recursive->isChecked())
			{
				configProjects(filename);
			}
			else if (fileinfo.fileName().endsWith(".vcxproj"))
			{
				if (!ui.SDK->isChecked())
				{
					configProject(filename);
				}
			}
			else if (fileinfo.fileName().endsWith(".sln"))
			{
				configSolution(filename);
			}
			else if (fileinfo.fileName().endsWith(".pro"))
			{
				configQtPro(filename);
			}
		}
	}
	else if (tmpinfo.isFile())
	{
		QFileInfo fileinfo = tmpinfo;
		QString filename = fileinfo.absoluteFilePath();
		if (fileinfo.fileName().endsWith(".vcxproj"))
		{
			if (!ui.SDK->isChecked())
			{
				configProject(filename);
			}
		}
		else if (fileinfo.fileName().endsWith(".sln"))
		{
			configSolution(filename);
		}
		else if (fileinfo.fileName().endsWith(".pro"))
		{
			configQtPro(filename);
		}
	}
}

void ConfigProject::configProject(QString projectname)
{
	QDomDocument * projectdoc=new QDomDocument();
	QFile file(projectname);
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
	QDomElement projectroot=projectdoc->documentElement();
	QDomElement propertygroup=projectroot.firstChildElement("PropertyGroup");

	while(propertygroup.attribute("Label")!="Globals")
	{
		propertygroup=propertygroup.nextSiblingElement("PropertyGroup");
	}
	if(propertygroup.isNull())
	{
		propertygroup=projectroot.appendChild(projectdoc->createElement("PropertyGroup")).toElement();
		propertygroup.setAttribute("Label","Globals");
	}
	propertygroup=propertygroup.nextSiblingElement("PropertyGroup");

	for(j=0;j<m;j++)
	{
		if(propertygroup.isNull())
		{
			propertygroup=projectroot.appendChild(projectdoc->createElement("PropertyGroup")).toElement();
		}
		else if(propertygroup.attribute("Label")=="UserMacros")
		{
			propertygroup=projectroot.insertBefore(projectdoc->createElement("PropertyGroup"),propertygroup).toElement();
		}
		propertygroup.setAttribute("Condition",conditions.at(j));
		setText(projectdoc,propertygroup,"PlatformToolset",ui.VSVersion->currentText());
		propertygroup=propertygroup.nextSiblingElement("PropertyGroup");
	}

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

	for(j=0;j<m;j++)
	{
		if(propertygroup.isNull())
		{
			propertygroup=projectroot.appendChild(projectdoc->createElement("PropertyGroup")).toElement();
		}
		propertygroup.setAttribute("Condition",conditions.at(j));
		if(ui.Library->isChecked())
		{
            setText(projectdoc,propertygroup,"OutDir","$(RobotSDK_SharedLibrary)\\");
		}
		else if(ui.Application->isChecked())
		{
            setText(projectdoc,propertygroup,"OutDir","$(SolutionDir)\\$(Configuration)\\");
		}
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
			itemdefine = itemdefine.nextSiblingElement("ItemDefinitionGroup");
			continue;
		}
		QDomElement predefineelem=itemdefine.firstChildElement("ClCompile").firstChildElement("PreprocessorDefinitions");
		QString predefine=predefineelem.text();

		if(ui.Library->isChecked())
		{
            if(!predefine.contains("RobotSDK_ModuleDev"))
			{
                predefine=QString("RobotSDK_ModuleDev;")+predefine;
			}
		}
		else
		{
            if(predefine.contains("RobotSDK_ModuleDev"))
			{
                predefine.remove(QString("RobotSDK_ModuleDev;"));
                predefine.remove(QString("RobotSDK_ModuleDev"));
			}
		}
		{
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
		if(ui.Library->isChecked())
		{
			QDomElement outputfileelem=itemdefine.firstChildElement("Link").firstChildElement("OutputFile");
			QString outputfile=outputfileelem.text();
            outputfile=QString("$(OutDir)$(ProjectName)_$(Configuration).dll");
			QDomNode tmpnode=outputfileelem.firstChild();
			while(!tmpnode.isNull()&&!tmpnode.isText())
			{
				tmpnode=tmpnode.nextSibling();
			}
			if(!tmpnode.isNull())
			{
				tmpnode.toText().setData(outputfile);
			}
		}
		itemdefine=itemdefine.nextSiblingElement("ItemDefinitionGroup");
	}

	file.open(QIODevice::WriteOnly | QIODevice::Text);
	QTextStream textstream(&file);
	projectdoc->save(textstream,2);
	file.close();
	delete projectdoc;
}

void ConfigProject::configSolution(QString solutionname)
{
	QFile file(solutionname);
	QStringList solution;
	if(file.open(QIODevice::ReadOnly | QIODevice::Text))
	{
		QTextStream stream(&file);
		QString tmpstr1=QString("Microsoft Visual Studio Solution File, Format Version ");
		QString tmpstr2=QString("# Visual Studio ");
		while(!stream.atEnd())
		{
			QString content=stream.readLine();
			if(content.startsWith(tmpstr1))
			{
				content=tmpstr1+formatversion.at(ui.VSVersion->currentIndex());
			}
			else if(content.startsWith(tmpstr2))
			{
				content=tmpstr2+vsversion.at(ui.VSVersion->currentIndex());
			}
			solution<<content;
		}
	}
	file.close();
	if(file.open(QIODevice::WriteOnly | QIODevice::Text))
	{
		QTextStream stream(&file);
		int i,n=solution.size();
		for(i=0;i<n;i++)
		{
			stream<<solution.at(i)<<"\n";
		}
	}
	file.close();
}

void ConfigProject::configQtPro(QString proname)
{
    QFile file;
    QTextStream stream;
    QString textcontent;
    QString filename=proname;
    QFileInfo fileinfo(filename);

    file.setFileName(filename);
    if(!file.open(QIODevice::ReadOnly|QIODevice::Text))
    {
		textcontent.clear();
    }
	else
	{
		textcontent = file.readAll();
		file.close();
	}    
    if(textcontent.contains("TEMPLATE = subdirs"))
    {
        return;
    }
    if(!textcontent.contains(QString("PROJNAME = %1").arg(fileinfo.baseName())))
    {
        textcontent=textcontent+QString("\nPROJNAME = %1").arg(fileinfo.baseName());
    }
    if(ui.Library->isChecked())
    {
        if(!textcontent.contains("INSTTYPE = MOD"))
        {
            textcontent=textcontent+QString("\nINSTTYPE = MOD");
        }
        if(textcontent.contains("INSTTYPE = APP"))
        {
            textcontent.remove(QString("\nINSTTYPE = APP"));
        }
        if(textcontent.contains("INSTTYPE = SDK"))
        {
            textcontent.remove(QString("\nINSTTYPE = SDK"));
        }
    }
    else if(ui.Application->isChecked())
    {
        if(!textcontent.contains("INSTTYPE = APP"))
        {
            textcontent=textcontent+QString("\nINSTTYPE = APP");
        }
        if(textcontent.contains("INSTTYPE = MOD"))
        {
            textcontent.remove(QString("\nINSTTYPE = MOD"));
        }
        if(textcontent.contains("INSTTYPE = SDK"))
        {
            textcontent.remove(QString("\nINSTTYPE = SDK"));
        }
    }
	else if (ui.SDK->isChecked())
	{
		if (!textcontent.contains("INSTTYPE = SDK"))
		{
			textcontent = textcontent + QString("\nINSTTYPE = SDK");
		}
		if (textcontent.contains("INSTTYPE = MOD"))
		{
			textcontent.remove(QString("\nINSTTYPE = MOD"));
		}
		if (textcontent.contains("INSTTYPE = APP"))
		{
			textcontent.remove(QString("\nINSTTYPE = APP"));
		}
	}
    if(!textcontent.contains("include(RobotSDK_Main.pri)"))
    {
        textcontent=textcontent+QString("\ninclude(RobotSDK_Main.pri)");
    }
	else
	{
		textcontent.remove(QString("\ninclude(RobotSDK_Main.pri)"));
		textcontent = textcontent + QString("\ninclude(RobotSDK_Main.pri)");
	}
    file.open(QIODevice::WriteOnly|QIODevice::Text);
    stream.setDevice(&file);
    stream<<textcontent;
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
