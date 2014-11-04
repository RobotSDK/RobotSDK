#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include <QtWidgets/QMainWindow>
#include "ui_configuration.h"

#include<qfile.h>
#include<qtextstream.h>
#include<qmessagebox.h>
#include<qboxlayout.h>

#include"registerdirwidget.h"

#define ConfigurationFile "configuration"

#define RobotSDK "RobotSDK"
#define Qt "Qt"
#define RobotDep "RobotDep"
#define Boost "Boost"

class Configuration : public QMainWindow
{
	Q_OBJECT

public:
	Configuration(int argc, char *argv[], QWidget *parent = 0);
	~Configuration();
private:
	Ui::ConfigurationClass ui;
protected:
	QString vsversion;
	QDomDocument * doc;
	RegisterDirWidget * robotsdkdir;
	RegisterDirWidget * robotdepdir;
public:
    bool addItem(QDomElement root, QString qstrRegisterName, QString qstrRegisterValue, bool addPath, QString qstrRegisterTag, QString qstrRegisterDir=QString(), bool force=0);
};

#endif // CONFIGURATION_H
