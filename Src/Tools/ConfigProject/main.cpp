#include "configproject.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	ConfigProject w;
	w.show();
	return a.exec();
}
