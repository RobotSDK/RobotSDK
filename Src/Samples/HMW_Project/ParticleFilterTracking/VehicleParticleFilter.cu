#include"VehicleParticleFilter.cuh"
#define SIZEMIN 1.0f
#define SIZEMAX 5.0f
#define SEARCHDIM 50

PARTICLE_INITIALIZE_FUNC(Vehicle,initialState,randomOffset)
{
    state.x=initialState.x+randomOffset.x;
    state.y=initialState.y+randomOffset.y;
    state.theta=initialState.theta+randomOffset.theta;
    state.v=initialState.v+randomOffset.v;
    state.width=initialState.width+randomOffset.width;
    state.length=initialState.length+randomOffset.length;
    state.wsigma=0;
    state.lsigma=0;
}

PARTICLE_RANDOMNIZE_FUNC(Vehicle,randomOffset)
{
    state.v+=randomOffset.v;
    state.theta+=randomOffset.theta;
}

PARTICLE_UPDATE_FUNC(Vehicle,deltaMsec)
{
    float msec=deltaMsec/1000.0;
    float dis=msec*state.v;
    state.x+=dis*cos(state.theta);
    state.y+=dis*sin(state.theta);
}

PARTICLE_TRANSFORM_FUNC(Vehicle,transform)
{
    transform.transformState2D(state.x,state.y,state.theta);
}

#define PI 3.14159265359
#define GETRECTCORNER(ctheta,stheta,cx,cy,x,y,corner,density,beamid) \
    corner[0]=x*ctheta-y*stheta+cx; \
    corner[1]=x*stheta+y*ctheta+cy; \
    beamid[0]=(atan2(corner[1],corner[0])+PI)/density; \
    corner[2]=x*ctheta+y*stheta+cx; \
    corner[3]=x*stheta-y*ctheta+cy; \
    beamid[1]=(atan2(corner[3],corner[2])+PI)/density; \
    corner[4]=-x*ctheta+y*stheta+cx; \
    corner[5]=-x*stheta-y*ctheta+cy; \
    beamid[2]=(atan2(corner[5],corner[4])+PI)/density; \
    corner[6]=-x*ctheta-y*stheta+cx; \
    corner[7]=-x*stheta+y*ctheta+cy; \
    beamid[3]=(atan2(corner[7],corner[6])+PI)/density; \
    corner[8]=x*ctheta-y*stheta+cx; \
    corner[9]=x*stheta+y*ctheta+cy; \
    beamid[4]=(atan2(corner[9],corner[8])+PI)/density;

#define GETRECTEDGE(ox,oy,x,y,edgeid) \
    if(ox>x) { \
        if(oy>y) { edgeid[0]=0;edgeid[1]=3; } \
        else if(oy<-y) { edgeid[0]=0;edgeid[1]=1; } \
        else { edgeid[0]=0;edgeid[1]=-1; } } \
    else if(ox<-x) { \
        if(oy>y) { edgeid[0]=2;edgeid[1]=3; } \
        else if(oy<-y) { edgeid[0]=2;edgeid[1]=1; } \
        else { edgeid[0]=2;edgeid[1]=-1; } } \
    else { \
        if(oy>y) { edgeid[0]=3;edgeid[1]=-1; } \
        else if(oy<-y) { edgeid[0]=1;edgeid[1]=-1; } \
        else { edgeid[0]=-1;edgeid[0]=-1; } }


PARTICLE_MEASURE_FUNC(Vehicle,state,measureData)
{
    float c=cos(state.theta);
    float s=sin(state.theta);

    float searchstep=SIZEMAX/(2*SEARCHDIM);

    float S[SEARCHDIM][SEARCHDIM];
    float corner[10];
    int beamid[5];
    int edgeid[2];

    float density=2*PI/measureData.beamnum;
    float thresh=0.05;

    float ox=-c*state.x-s*state.y;
    float oy=s*state.x-c*state.y;
    float weight=-1;
    int maxi,maxj;
    float sumweight=0;
    for(int i=0;i<SEARCHDIM;i++)
    {
        float x=(i+1)*searchstep;
        for(int j=0;j<SEARCHDIM;j++)
        {
            float y=(j+1)*searchstep;
            GETRECTCORNER(c,s,state.x,state.y,x,y,corner,density,beamid);
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
                        score*=0.5+(expf(-powf(distance-thresh,2)/0.01))*9.5;
                        //score*=0.5f;
                    }
                    else if(distance<-thresh)
                    {
                        score*=0.1+(expf(-powf(distance+thresh,2)/0.01))*9.9;
                        //score*=0.1f;
                    }
                    else
                    {
                        score*=10+(expf(-powf(distance,2)/0.01))*10;
                        //score*=10.0f;
                    }
                    count++;
                }
                if(count>0)
                {
                    S[i][j]*=powf(score,1.0f/count);
                }
                else
                {
                    S[i][j]=0;
                    break;
                }
            }
            sumweight+=S[i][j];
            if(weight<S[i][j])
            {
                weight=S[i][j];
                maxi=i;
                maxj=j;
            }
        }
    }
    if(weight<=0||sumweight==0)
    {
        return 0;
    }
    weight/=sumweight;
    float width=(maxj+1)*searchstep*2;
    float length=(maxi+1)*searchstep*2;
    float wsigma=0;
    float wsum=0;
    float lsigma=0;
    float lsum=0;
    for(int k=0;k<SEARCHDIM;k++)
    {
        float wdis=(maxj-k)*searchstep*2;
        wsigma+=S[maxi][k]/sumweight*wdis*wdis;
        wsum+=S[maxi][k]/sumweight;
        float ldis=(maxi-k)*searchstep*2;
        lsigma+=S[k][maxj]/sumweight*ldis*ldis;
        lsum+=S[k][maxj]/sumweight;
    }
    wsigma/=wsum;
    lsigma/=lsum;
    if(state.lsigma*state.wsigma==0)
    {
        state.width=width;
        state.length=length;
        state.wsigma=wsigma;
        state.lsigma=lsigma;
        return weight;
    }
    float tmpwsigma=wsigma*state.wsigma/(wsigma+state.wsigma);
    float tmplsigma=lsigma*state.lsigma/(lsigma+state.lsigma);

    weight*=sqrt((tmpwsigma*tmplsigma)/(state.wsigma*state.lsigma));

    float wdis=width-state.width;
    float ldis=length-state.length;
    weight*=expf(-(wdis*wdis)/(wsigma+state.wsigma)-(ldis*ldis)/(lsigma+state.lsigma));

    state.width=width;
    state.length=length;
    state.wsigma=tmpwsigma;
    state.lsigma=tmplsigma;

    return weight;
}

PARTICLE_FILTER_MEASURE_FUNC(Vehicle,measureData)
{
    MEASUREDATA_TYPE(Vehicle) virtualscan;
    virtualscan.beamnum=measureData.beamnum;
    size_t beamsize=virtualscan.beamnum*sizeof(double);
    cudaMalloc((void **)(&(virtualscan.beams)),beamsize);
    cudaMemcpy(virtualscan.beams,measureData.beams,beamsize,cudaMemcpyHostToDevice);
    Vehicle_Base::measureParticles(virtualscan);
    cudaDeviceSynchronize();
    cudaFree(virtualscan.beams);
}

PARTICLE_FILTER_INTERACT_FUNCS(Vehicle)
