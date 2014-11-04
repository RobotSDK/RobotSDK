#ifndef CONFIGPROJECT_H
#define CONFIGPROJECT_H

#include <QtWidgets/QMainWindow>
#include "ui_configproject.h"

#include<qdom.h>
#include<qstring.h>
#include<qlist.h>
#include<qfiledialog.h>
#include<qfile.h>
#include<qtextstream.h>
#include<qmessagebox.h>
#include<qdatetime.h>

#ifdef Q_OS_LINUX
    #define ROBOTSDKKERNEL QString("%1/SDK/RobotSDK/Kernel").arg(QString(qgetenv("HOME")))
    #define ROBOTSDKMODULEDEV QString("%1/SDK/RobotSDK/ModuleDev").arg(QString(qgetenv("HOME")))
    #define ROBOTSDKMODULE QString("%1/SDK/RobotSDK/Module").arg(QString(qgetenv("HOME")))
    #define ROBOTSDKSHAREDLIBRARY QString("%1/SDK/RobotSDK/Module/SharedLibrary").arg(QString(qgetenv("HOME")))
    #define ROBOTSDKTOOLS QString("%1/SDK/RobotSDK/Tools").arg(QString(qgetenv("HOME")))
#elif defined(Q_OS_WIN)
    #define ROBOTSDKKERNEL getenv("RobotSDK_Kernel")
    #define ROBOTSDKMODULEDEV getenv("RobotSDK_ModuleDev")
    #define ROBOTSDKMODULE getenv("RobotSDK_Module")
    #define ROBOTSDKSHAREDLIBRARY getenv("RobotSDK_SharedLibrary")
	#define ROBOTSDKTOOLS getenv("RobotSDK_Tools")
#endif

class ConfigProject : public QMainWindow
{
	Q_OBJECT

public:
	ConfigProject(QWidget *parent = 0);
	~ConfigProject();
private:
	Ui::ConfigProjectClass ui;
protected:
	QStringList vversion;
	QStringList formatversion;
	QStringList vsversion;
public slots:
	void browseSlot();
	void configSlot();
protected:
	void setText(QDomDocument * tmpdoc, QDomElement & tmproot, QString tag, QString text);
	void configProjects(QString projectsdir);
	void configProject(QString projectname);
	void configSolution(QString solutionname);
    void configQtPro(QString proname);
};

#endif // CONFIGPROJECT_H
