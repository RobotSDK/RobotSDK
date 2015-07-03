#include"VehicleParticleFilter.cuh"
#define SIZEMIN 1.0f
#define SIZEMAX 5.0f
#define SEARCHDIM 50

PARTICLE_INITIALIZE_FUNC(Vehicle,initialState,randomOffset)
{
    state.x=initialState.x+randomOffset.x;
    state.y=initialState.y+randomOffset.y;
    state.theta=initialState.theta+randomOffset.theta;
    state.thetaoffset=randomOffset.thetaoffset;
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
    state.theta+=state.thetaoffset;
}

PARTICLE_TRANSFORM_FUNC(Vehicle,transform)
{
    transform.transformState2D(state.x,state.y,state.theta);
}

#define GETWEIGHT(ctheta,stheta,statex,statey,mapsize,gridsize,obmap,obmapdata,radius,wtable,sigma,x,y,weight,innerscore,flag) \
    if(weight>0) \
    { \
        float gx=ctheta*x-stheta*y+statex; \
        float gy=stheta*x+ctheta*y+statey; \
        int mx=mapsize/2-int(gy/gridsize+0.5f); \
        int my=mapsize/2-int(gx/gridsize+0.5f); \
        float dis=sqrt(gx*gx+gy*gy); \
        if(mx>=0&&mx<mapsize&&my>=0&&my<mapsize) \
        { \
            flag=0; \
            int mapid=(my*mapsize+mx)*3; \
            int mapdataid=my*mapsize+mx; \
            if(obmap[mapid+0]==255) \
            { \
                float delta=obmapdata[mapdataid]*radius; \
                weight*=wtable[0]/2+(exp(-delta*delta/sigma))*(wtable[0]/2); \
                innerscore *=wtable[2]; \
            } \
            else if(obmap[mapid+1]==255) \
            { \
                float delta=obmapdata[mapdataid]*dis; \
                weight*=wtable[2]+(exp(-delta*delta/sigma))*(wtable[0]/2-wtable[2]); \
                innerscore *=wtable[2]; \
            } \
            else \
            { \
                float delta=obmapdata[mapdataid]*dis; \
                weight*=wtable[1]+(exp(-delta*delta/sigma))*(wtable[0]/2-wtable[1]); \
            } \
        } \
        else \
        { \
            weight=0; \
        } \
    }

#define CALINSIDEWEIGHT(ctheta,stheta,statex,statey,mapsize,gridsize,obmap,obmapdata,radius,wtable,sigma,x,y,weight,flag) \


PARTICLE_MEASURE_FUNC(Vehicle,state,measureData)
{
    float c=cos(state.theta);
    float s=sin(state.theta);

    float searchstep=SIZEMAX/(2*SEARCHDIM);

    float V[SEARCHDIM][SEARCHDIM];
    float W[SEARCHDIM][SEARCHDIM];
    float L[SEARCHDIM][SEARCHDIM];
    float I[SEARCHDIM][SEARCHDIM];
    float S[SEARCHDIM][SEARCHDIM];

    bool flag=1;
    for(int i=0;i<SEARCHDIM;i++)
    {
        float x=i*searchstep;
        for(int j=0;j<SEARCHDIM;j++)
        {
            float y=j*searchstep;
            V[i][j]=1.0f;
            float tmpinnerscore=1.0f;
            GETWEIGHT(c,s,state.x,state.y,measureData.mapsize,measureData.gridsize,measureData.map,measureData.mapdata,measureData.radius,measureData.wtable,measureData.sigma,x,y,V[i][j],tmpinnerscore,flag);
            GETWEIGHT(c,s,state.x,state.y,measureData.mapsize,measureData.gridsize,measureData.map,measureData.mapdata,measureData.radius,measureData.wtable,measureData.sigma,x,-y,V[i][j],tmpinnerscore,flag);
            GETWEIGHT(c,s,state.x,state.y,measureData.mapsize,measureData.gridsize,measureData.map,measureData.mapdata,measureData.radius,measureData.wtable,measureData.sigma,-x,y,V[i][j],tmpinnerscore,flag);
            GETWEIGHT(c,s,state.x,state.y,measureData.mapsize,measureData.gridsize,measureData.map,measureData.mapdata,measureData.radius,measureData.wtable,measureData.sigma,-x,-y,V[i][j],tmpinnerscore,flag);
            if(j>0)
            {
                I[i][j]=I[i][j-1]*tmpinnerscore;
            }
            else
            {
                I[i][j]=tmpinnerscore;
            }
        }
    }
    if(flag)
    {
        return 0;
    }
    for(int i=0;i<SEARCHDIM;i++)
    {
        W[i][0]=V[i][0];
        L[0][i]=V[0][i];
        for(int j=1;j<SEARCHDIM;j++)
        {
            W[i][j]=W[i][j-1]*V[i][j];
            L[j][i]=L[j-1][i]*V[j][i];
        }
    }
    int startid=int(SIZEMIN/(2*searchstep));
    float weight=-1;
    int maxi,maxj;
    for(int i=startid;i<SEARCHDIM;i++)
    {
        for(int j=startid;j<SEARCHDIM;j++)
        {
            S[i][j]=W[i][j]*L[i][j];
            if(i>0&&j>0)
            {
                S[i][j]*=I[i-1][j-1];
            }
            if(weight<S[i][j])
            {
                weight=S[i][j];
                maxi=i;
                maxj=j;
            }
        }
    }
    if(weight<=0)
    {
        return 0;
    }
    float width=maxj*searchstep*2;
    float length=maxi*searchstep*2;
    float wsigma=0;
    float wsum=0;
    float lsigma=0;
    float lsum=0;
    for(int k=startid;k<SEARCHDIM;k++)
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
    weight*=exp(-(wdis*wdis)/(wsigma+state.wsigma)-(ldis*ldis)/(lsigma+state.lsigma));

    state.width=width;
    state.length=length;
    state.wsigma=tmpwsigma;
    state.lsigma=tmplsigma;

    return weight;
}

PARTICLE_FILTER_MEASURE_FUNC(Vehicle, measureData)
{
    MEASUREDATA_TYPE(Vehicle) obstaclemap=measureData;
    size_t mapsize=obstaclemap.mapsize*obstaclemap.mapsize*3*sizeof(unsigned char);
    cudaMalloc((void **)(&(obstaclemap.map)),mapsize);
    cudaMemcpy(obstaclemap.map,measureData.map,mapsize,cudaMemcpyHostToDevice);
    size_t mapdatasize=obstaclemap.mapsize*obstaclemap.mapsize*sizeof(float);
    cudaMalloc((void **)(&(obstaclemap.mapdata)),mapdatasize);
    cudaMemcpy(obstaclemap.mapdata,measureData.mapdata,mapdatasize,cudaMemcpyHostToDevice);
    Vehicle_Base::measureParticles(obstaclemap);
    cudaDeviceSynchronize();
    cudaFree(obstaclemap.map);
    cudaFree(obstaclemap.mapdata);
}

PARTICLE_FILTER_INTERACT_FUNCS(Vehicle)
