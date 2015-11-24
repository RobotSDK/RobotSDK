#include <QCoreApplication>

#include"particlefilter.h"

#include<iostream>
#include<chrono>
using namespace std::chrono;

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    STATE_TYPE(Vehicle) statemin={-1,-1,-0.1,-5,-2,-2};
    STATE_TYPE(Vehicle) statemax={1,1,0.1,5,2,2};
    STATE_TYPE(Vehicle) statemean={0,0,0.0,0,0};
    STATE_TYPE(Vehicle) statesigma={0,0,0.2,10,1,1};

    PF_Vehicle_initialParticleFilter(5000,statemin,statemax,statemean,statesigma);

    MEASUREDATA_TYPE(Vehicle) target;
    target.state={0,0,0,1,2.5,3.5};

    srand(time(NULL));
    int objectnumhalf=100;
    for(int i=-objectnumhalf;i<=objectnumhalf;i++)
    {
        STATE_TYPE(Vehicle) state={target.state.x+(rand()%10/10.0f)*i,target.state.y+(rand()%10/10.0f)*i,target.state.theta+(rand()%10/100.0f)*i
                ,target.state.v+(rand()%10/10.0f)*i,target.state.width+(rand()%10/100.0f)*i,target.state.length+(rand()%10/100.0f)*i};
        PF_Vehicle_addObjectState(i,state);
    }


    std::vector<STATE_TYPE(Vehicle)> states;
    std::vector<int> objectid;
    int msec=1000;
    float threshold=0.99;

    for(int j=0;j<100;j++)
    {
        float dis=target.state.v*msec/1000.0;
        target.state.v+=rand()%10/10.0;
        target.state.theta+=rand()%10/100.0;
        target.state.x+=dis*cos(target.state.theta);
        target.state.y+=dis*sin(target.state.theta);

        std::cout<<"========================================\n";
        std::cout<<"Target\n";
        std::cout<<j<<"\t"<<target.state.x<<"\t"<<target.state.y<<"\t"<<target.state.theta<<"\t"<<target.state.v<<"\t"<<target.state.width<<"\t"<<target.state.length<<"\n";

        milliseconds start,end;
        std::chrono::duration<double> elapsed_seconds;

        start = duration_cast< milliseconds >(system_clock::now().time_since_epoch());
        PF_Vehicle_advanceParticleFilter2D(msec,target);
        PF_Vehicle_removeParticles(threshold);
        PF_Vehicle_estimateObjects(objectid,states);
        end = duration_cast< milliseconds >(system_clock::now().time_since_epoch());

        elapsed_seconds = end-start;
        std::cout<<"PF Time Cost:"<<elapsed_seconds.count()<<" s\n";

        std::cout<<"Estimates:\n";
        int n=states.size();
        for(int i=0;i<n;i++)
        {
            std::cout<<objectid[i]<<"\t"<<states[i].x<<"\t"<<states[i].y<<"\t"<<states[i].theta<<"\t"<<states[i].v<<"\t"<<states[i].width<<"\t"<<states[i].length<<"\n";
        }
        //getchar();
    }
    return a.exec();
}
