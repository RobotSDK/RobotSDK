#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include<RobotSDK_Global.h>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
public:
    RobotSDK::Graph graph;
};

#endif // MAINWINDOW_H
