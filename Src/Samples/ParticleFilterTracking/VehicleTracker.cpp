#include"VehicleTracker.h"
using namespace RobotSDK_Module;

//If you need to use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Uncomment below PORT_DECL and set input node class name
PORT_DECL(0, VehicleDetector)
PORT_DECL(1, VirtualScanGlobalizer)

//=================================================
//Original node functions

//If you don't need to manually open node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, openNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    STATE_TYPE(Vehicle) statemin={vars->state_x_min,vars->state_y_min,vars->state_theta_min,vars->state_v_min,vars->state_width_min,vars->state_length_min};
    STATE_TYPE(Vehicle) statemax={vars->state_x_max,vars->state_y_max,vars->state_theta_max,vars->state_v_max,vars->state_width_max,vars->state_length_max};
    STATE_TYPE(Vehicle) statemean={0,0,0,0,0,0};
    STATE_TYPE(Vehicle) statesigma={0,0,vars->state_theta_sigma,vars->state_v_sigma,0,0};
    PF_Vehicle_initialParticleFilter(vars->particlenum,statemin,statemax,statemean,statesigma);

    vars->sync.clear();

    vars->curtimestamp=QTime();
    vars->curtransform=cv::Mat::eye(4,4,CV_64F);
    vars->curtheta=0;
    vars->objectid.clear();
    vars->objectstate.clear();

    vars->localheadvec=cv::Mat::zeros(4,1,CV_64F);
    vars->localheadvec.at<double>(0)=1;

    QVector<double> GLB(2);
    QVector<double> GUB(2);
    QVector<double> GStep(2);
    GLB[0]=1;GLB[1]=1;
    GUB[0]=5;GUB[1]=5;
    GStep[0]=0.1;GStep[1]=0.1;
    vars->fitter=new FastRectangleFitting(GLB,GUB,GStep);

	return 1;
}

//If you don't need to manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    PF_Vehicle_clear();

    vars->sync.clear();

    vars->objectid.clear();
    vars->objectstate.clear();

    delete vars->fitter;

	return 1;
}

#define SIZEMIN 1.0f
#define SIZEMAX 5.0f
#define SEARCHDIM 50
#define PI 3.14159265359

void GETRECTCORNER(float ctheta, float stheta, float cx, float cy, float x, float y, float * corner, float density, int * beamid)
{
    corner[0]=x*ctheta-y*stheta+cx;
    corner[1]=x*stheta+y*ctheta+cy;
    beamid[0]=(atan2(corner[1],corner[0])+PI)/density;
    corner[2]=x*ctheta+y*stheta+cx;
    corner[3]=x*stheta-y*ctheta+cy;
    beamid[1]=(atan2(corner[3],corner[2])+PI)/density;
    corner[4]=-x*ctheta+y*stheta+cx;
    corner[5]=-x*stheta-y*ctheta+cy;
    beamid[2]=(atan2(corner[5],corner[4])+PI)/density;
    corner[6]=-x*ctheta-y*stheta+cx;
    corner[7]=-x*stheta+y*ctheta+cy;
    beamid[3]=(atan2(corner[7],corner[6])+PI)/density;
    corner[8]=x*ctheta-y*stheta+cx;
    corner[9]=x*stheta+y*ctheta+cy;
    beamid[4]=(atan2(corner[9],corner[8])+PI)/density;
}

void GETRECTEDGE(float ox, float oy, float x, float y, int * edgeid)
{
    if(ox>x) {
        if(oy>y) { edgeid[0]=0;edgeid[1]=3; }
        else if(oy<-y) { edgeid[0]=0;edgeid[1]=1; }
        else { edgeid[0]=0;edgeid[1]=-1; } }
    else if(ox<-x) {
        if(oy>y) { edgeid[0]=2;edgeid[1]=3; }
        else if(oy<-y) { edgeid[0]=2;edgeid[1]=1; }
        else { edgeid[0]=2;edgeid[1]=-1; } }
    else {
        if(oy>y) { edgeid[0]=3;edgeid[1]=-1; }
        else if(oy<-y) { edgeid[0]=1;edgeid[1]=-1; }
        else { edgeid[0]=-1;edgeid[0]=-1; } }
}

void measureParticles(std::vector< std::vector<STATE_TYPE(Vehicle)> > & particleState, std::vector< std::vector<float> > & particleWeight, MEASUREDATA_TYPE(Vehicle) & measureData)
{
    for(int ii=0;ii<particleState.size();ii++)
    {
        for(int jj=0;jj<particleState[ii].size();jj++)
        {
            float c=cos(particleState[ii][jj].theta);
            float s=sin(particleState[ii][jj].theta);

            float searchstep=SIZEMAX/(2*SEARCHDIM);

            float S[SEARCHDIM][SEARCHDIM];
            float corner[10];
            int beamid[5];
            int edgeid[2];

            float density=2*PI/measureData.beamnum;
            float thresh=0.1;

            float ox=-c*particleState[ii][jj].x-s*particleState[ii][jj].y;
            float oy=s*particleState[ii][jj].x-c*particleState[ii][jj].y;
            float weight=-1;
            int maxi,maxj;
            float sumweight=0;
            for(int i=0;i<SEARCHDIM;i++)
            {
                float x=(i+1)*searchstep;
                for(int j=0;j<SEARCHDIM;j++)
                {
                    float y=(j+1)*searchstep;
                    GETRECTCORNER(c,s,particleState[ii][jj].x,particleState[ii][jj].y,x,y,corner,density,beamid);
                    GETRECTEDGE(ox,oy,x,y,edgeid);                    
                    S[i][j]=1.0f;
                    for(int k=0;k<2;k++)
                    {
                        if(edgeid[k]<0)
                        {
                            break;
                        }
                        int count=0;
                        float score=1;
                        int startid=beamid[edgeid[k]];
                        int endid=beamid[edgeid[k]+1];
                        if(startid>endid)
                        {
                            endid+=measureData.beamnum;
                        }
                        for(int id=startid;id<=endid;id++)
                        {
                            int tmpid=id%measureData.beamnum;
                            if(measureData.beams[tmpid]<=0)
                            {
                                continue;
                            }
                            float theta=id*density-PI;
                            float bx=measureData.beams[tmpid]*cos(theta);
                            float by=measureData.beams[tmpid]*sin(theta);
                            float beta=(corner[2*edgeid[k]+1]*corner[2*edgeid[k]+2]-corner[2*edgeid[k]]*corner[2*edgeid[k]+3])
                                    /(by*(corner[2*edgeid[k]+2]-corner[2*edgeid[k]])-bx*(corner[2*edgeid[k]+3]-corner[2*edgeid[k]+1]));
                            float distance=(beta-1)*measureData.beams[tmpid];
                            if(distance>thresh)
                            {
                                score*=0.5+(exp(-pow(distance-thresh,2)/0.01))*9.5;
                                //score*=0.5f;
                            }
                            else if(distance<-thresh)
                            {
                                score*=0.1+(exp(-pow(distance+thresh,2)/0.01))*9.9;
                                //score*=0.1f;
                            }
                            else
                            {
                                score*=10+(exp(-pow(distance,2)/0.01))*10;
                                //score*=10.0f;
                            }
                            count++;
                        }
                        if(count>0)
                        {
                            S[i][j]*=pow(score,1.0/count);
                        }
                        else
                        {
                            S[i][j]=0;
                            break;
                        }
                    }
                    sumweight+=S[i][j];
                    if(weight<=S[i][j])
                    {
                        weight=S[i][j];
                        maxi=i;
                        maxj=j;
                    }
                }
            }
            if(weight<=0||sumweight==0)
            {
                particleWeight[ii][jj]=0;
                continue;
            }
            //weight/=sumweight;
            float width=(maxj+1)*searchstep*2;
            float length=(maxi+1)*searchstep*2;
            float wsigma=0;
            float wsum=0;
            float lsigma=0;
            float lsum=0;
            for(int k=0;k<SEARCHDIM;k++)
            {
                float wdis=(maxj-k)*searchstep*2;
                wsigma+=S[maxi][k]*wdis*wdis;
                wsum+=S[maxi][k];
                float ldis=(maxi-k)*searchstep*2;
                lsigma+=S[k][maxj]*ldis*ldis;
                lsum+=S[k][maxj];
            }
            wsigma/=wsum;
            lsigma/=lsum;
            if(particleState[ii][jj].lsigma*particleState[ii][jj].wsigma==0)
            {
                particleState[ii][jj].width=width;
                particleState[ii][jj].length=length;
                particleState[ii][jj].wsigma=wsigma;
                particleState[ii][jj].lsigma=lsigma;
                particleWeight[ii][jj]=weight;
                continue;
            }
            float tmpwsigma=wsigma*particleState[ii][jj].wsigma/(wsigma+particleState[ii][jj].wsigma);
            float tmplsigma=lsigma*particleState[ii][jj].lsigma/(lsigma+particleState[ii][jj].lsigma);

            weight*=sqrt((tmpwsigma*tmplsigma)/(particleState[ii][jj].wsigma*particleState[ii][jj].lsigma));

            float wdis=width-particleState[ii][jj].width;
            float ldis=length-particleState[ii][jj].length;
            weight*=expf(-(wdis*wdis)/(wsigma+particleState[ii][jj].wsigma)-(ldis*ldis)/(lsigma+particleState[ii][jj].lsigma));

            particleState[ii][jj].width=width;
            particleState[ii][jj].length=length;
            particleState[ii][jj].wsigma=tmpwsigma;
            particleState[ii][jj].lsigma=tmplsigma;
            particleWeight[ii][jj]=weight;
        }
    }
}

NODE_FUNC_DEF(void, measureParticlesFastFitting, std::vector< std::vector<STATE_TYPE(Vehicle)> > & particleState, std::vector< std::vector<float> > & particleWeight)
{
    auto vars=NODE_VARS;
    for(int ii=0;ii<particleState.size();ii++)
    {
        for(int jj=0;jj<particleState[ii].size();jj++)
        {
            vars->fitter->updatePosition(particleState[ii][jj].x,particleState[ii][jj].y);
            vars->fitter->updateOrientation(particleState[ii][jj].theta);

            Geometry preG;
            preG.geometry.resize(2);
            preG.geometry[0]=particleState[ii][jj].length;
            preG.geometry[1]=particleState[ii][jj].width;

            preG.sigma.resize(2);
            preG.sigma[0]=particleState[ii][jj].lsigma;
            preG.sigma[1]=particleState[ii][jj].wsigma;

            preG.score=particleWeight[ii][jj];

            Geometry tmpG;
            vars->fitter->getFitting(tmpG);
            particleState[ii][jj].length=tmpG.geometry[0];
            particleState[ii][jj].width=tmpG.geometry[1];
            particleWeight[ii][jj]=tmpG.score;

            if(preG.sigma[0]*preG.sigma[1]==0)
            {
                particleState[ii][jj].lsigma=tmpG.sigma[0];
                particleState[ii][jj].wsigma=tmpG.sigma[1];
            }
            else
            {
                particleState[ii][jj].lsigma=(tmpG.sigma[0]*preG.sigma[0])/(tmpG.sigma[0]+preG.sigma[0]);
                particleState[ii][jj].wsigma=(tmpG.sigma[1]*preG.sigma[1])/(tmpG.sigma[1]+preG.sigma[1]);
                particleWeight[ii][jj]*=sqrt(particleState[ii][jj].lsigma/preG.sigma[0])*exp(-pow(tmpG.geometry[0]-preG.geometry[0],2)/(tmpG.sigma[0]+preG.sigma[0]));
                particleWeight[ii][jj]*=sqrt(particleState[ii][jj].wsigma/preG.sigma[1])*exp(-pow(tmpG.geometry[1]-preG.geometry[1],2)/(tmpG.sigma[1]+preG.sigma[1]));
            }
        }
    }
}

#define DEBUGPF

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    if(SYNC_START(vars->sync))
    {
        auto detection=SYNC_DATA(vars->sync,0);
        auto vscan=SYNC_DATA(vars->sync,1);
        cv::Mat head=vars->curtransform*vars->localheadvec;
        if(vars->curtimestamp.isNull())
        {
            vars->curtimestamp=vscan->timestamp;
            vars->curtransform=vscan->egotransform;
            vars->curtheta=atan2(head.at<double>(1),head.at<double>(0));
            vars->measuredata.beamnum=vscan->virtualscan.size();
            vars->measuredata.beams=new double[vars->measuredata.beamnum];
            PF_Vehicle_addObjectStates(detection->objectids,detection->objectstates);
            return 0;
        }
        else
        {
            int deltamsec=vars->curtimestamp.msecsTo(vscan->timestamp);
            vars->measuredata.beamnum=vscan->virtualscan.size();
            int datasize=vars->measuredata.beamnum*sizeof(double);
            vars->measuredata.beams=new double[vars->measuredata.beamnum];
            memcpy(vars->measuredata.beams,vscan->virtualscan.data(),datasize);
            float dx=vscan->egotransform.at<double>(0,3)-vars->curtransform.at<double>(0,3);
            float dy=vscan->egotransform.at<double>(1,3)-vars->curtransform.at<double>(1,3);
            float theta1=vars->curtheta;
            float theta2=atan2(head.at<double>(1),head.at<double>(0));

            std::vector< std::vector<STATE_TYPE(Vehicle)> > particlestate;
            std::vector< std::vector<float> > particleweight;

            PF_Vehicle_randomnizeParticles();
            PF_Vehicle_debugParticleFilterGetParticles(particlestate,particleweight);

            PF_Vehicle_updateParticles(deltamsec);
            PF_Vehicle_debugParticleFilterGetParticles(particlestate,particleweight);

            PF_Vehicle_transformParticles2D(dx,dy,theta1,theta2);
            PF_Vehicle_debugParticleFilterGetParticles(particlestate,particleweight);

            PF_Vehicle_measureParticles(vars->measuredata);
            PF_Vehicle_debugParticleFilterGetParticles(particlestate,particleweight);

//            measureParticles(particlestate,particleweight,vars->measuredata);
//            PF_Vehicle_debugParticleFilterSetParticles(particlestate,particleweight);

            QVector<double> virtualscan=vscan->virtualscan;
            vars->fitter->updateScanBeams(virtualscan);
//            NODE_FUNC(measureParticlesFastFitting,particlestate,particleweight);
//            PF_Vehicle_debugParticleFilterSetParticles(particlestate,particleweight);

            PF_Vehicle_resampleParticles();
            PF_Vehicle_debugParticleFilterGetParticles(particlestate,particleweight);

            //PF_Vehicle_advanceParticleFilter2D(deltamsec,vars->measuredata,dx,dy,theta1,theta2);

//            PF_Vehicle_removeParticles(vars->threshold);
//            PF_Vehicle_debugParticleFilterGetParticles(particlestate,particleweight);

            PF_Vehicle_estimateObjects(vars->objectid,vars->objectstate);
            PF_Vehicle_debugParticleFilterGetParticles(particlestate,particleweight);

            vars->curtimestamp=vscan->timestamp;
            vars->curtransform=vscan->egotransform;
            vars->curtheta=theta2;
            PF_Vehicle_addObjectStates(detection->objectids,detection->objectstates);

            auto data=NODE_DATA;
            data->timestamp=vars->curtimestamp;
            data->transform=vars->curtransform.clone();

#ifdef DEBUGPF
            std::vector<int> ids;
            if(particlestate.size()>0)
            {
                ids.resize(particlestate[0].size(),0);
                data->objectid=ids;
                data->objectstate=particlestate[0];
                data->weights=particleweight[0];
            }
#else
            data->objectid=vars->objectid;
            data->objectstate=vars->objectstate;
            data->weights.resize(data->objectid.size(),1.0/vars->particlenum);
#endif
            delete [](vars->measuredata.beams);

            return 1;
        }
    }
    return 0;
}
