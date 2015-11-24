#include"VirtualScanCluster.h"
using namespace RobotSDK_Module;

//If you need use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Uncomment below PORT_DECL and set input node class name
PORT_DECL(0, VirtualScanGenerator)

//=================================================
//Original node functions

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
    NOUNUSEDWARNING;
    auto params=NODE_PARAMS;
    auto outputdata=NODE_DATA;
    auto inputparams=PORT_PARAMS(0,0);
    auto data=PORT_DATA(0,0);

    double PI=3.141592654;
    double density=2*PI/inputparams->beamnum;

    outputdata->timestamp=data->timestamp;
    outputdata->extrinsicmat=data->extrinsicmat.clone();
    outputdata->rospoints=data->rospoints;
    outputdata->virtualscan=data->virtualscan;
    outputdata->minheights=data->minheights;
    outputdata->maxheights=data->maxheights;
    outputdata->labels=data->labels;
    outputdata->labels.fill(0);
    outputdata->clusternum=0;
    outputdata->clusters.clear();

    uint i,n=outputdata->virtualscan.size();
    for(i=0;i<n;i++)
    {
        if(outputdata->virtualscan[i]>inputparams->minrange&&outputdata->labels[i]==0)
        {
            QQueue<uint> queue;
            queue.push_back(i);
            outputdata->clusternum++;
            outputdata->labels[i]=outputdata->clusternum;
            QVector<uint> records;
            while(!queue.isEmpty())
            {
                uint id=queue.front();
                queue.pop_front();
                records.push_back(id);

                int j,neighbornum=params->neighbornum;
                for(j=-neighbornum;j<=neighbornum;j++)
                {
                    if(j==0)
                    {
                        continue;
                    }
                    int nid=int(id)+j;
                    if(nid<0)
                    {
                        nid+=n;
                    }
                    if(nid>=int(n))
                    {
                        nid-=n;
                    }
                    if(outputdata->virtualscan[nid]>0&&outputdata->labels[nid]==0)
                    {
                        double angle=fabs(j*density);
                        double xsigma=params->xsigma*outputdata->virtualscan[id];
                        if(xsigma<params->xminsigma)
                        {
                            xsigma=params->xminsigma;
                        }
                        double xdis=exp(-pow(outputdata->virtualscan[nid]*sin(angle),2)/(2*pow(xsigma,2)));
                        double ysigma=params->ysigma*outputdata->virtualscan[id];
                        if(ysigma<params->yminsigma)
                        {
                            ysigma=params->yminsigma;
                        }
                        double ydis=exp(-pow(outputdata->virtualscan[nid]*cos(angle)-outputdata->virtualscan[id],2)/(2*pow(ysigma,2)));
                        double dis=xdis*ydis;
                        if(dis>params->threshold)
                        {
                            outputdata->labels[nid]=outputdata->clusternum;
                            queue.push_back(nid);
                        }
                    }
                }
            }
            if(records.size()<int(params->minpointsnum))
            {
                int j,m=records.size();
                for(j=0;j<m;j++)
                {
                    outputdata->labels[records[j]]=0;
                }
                outputdata->clusternum--;
            }
        }
    }
    for(i=0;i<n;i++)
    {
        outputdata->clusters.insert(outputdata->labels[i],i);
    }
    return 1;
}
