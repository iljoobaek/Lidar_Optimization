#include "fastvirtualscan/fastvirtualscan.h"
#include "../../timers.h"
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

/*
rm 5_openmp_cuda_ds_change; rm -rf build; mkdir build; cd build; cmake ..; make -j8; cd ..; ./5_openmp_cuda_ds_change;
*/

#define MAXVIRTUALSCAN 1e6
#define SIZE 160
#define USEOMP

extern Timers timers;

FastVirtualScan::FastVirtualScan(int beamNum, double heightStep, double minFloor, double maxCeiling)
{
    // beamnum=1000;
    // step=0.3;
    // minfloor=-3;
    // maxceiling=3;
    beamnum=beamNum;
    step=heightStep;
    minfloor=minFloor;
    maxceiling=maxCeiling;
    rotation=0;
    minrange=0;
    
    int size=int((maxceiling-minfloor)/step+0.5); // 160
    std::cout << "cudaMallocManaged ret: " << cudaMallocManaged(&svs, beamnum*size*sizeof(SimpleVirtualScan)) << std::endl;
    std::cout << "cudaMallocManaged ret: " << cudaMallocManaged(&svsback, beamnum*size*sizeof(SimpleVirtualScan)) << std::endl;
}

FastVirtualScan::~FastVirtualScan()
{
    cudaFree(svs);
    cudaFree(svsback);
}

__device__
bool compareDistance(const SimpleVirtualScan & svs1, const SimpleVirtualScan & svs2)
{
    if(svs1.rotlength==svs2.rotlength)
    {
        return svs1.rotid>svs2.rotid;
    }
    else
    {
        return svs1.rotlength<svs2.rotlength;
    }
}
bool compareDistanceHost(const SimpleVirtualScan & svs1, const SimpleVirtualScan & svs2)
{
    if(svs1.rotlength==svs2.rotlength)
    {
        return svs1.rotid>svs2.rotid;
    }
    else
    {
        return svs1.rotlength<svs2.rotlength;
    }
}



__global__
void init_simple_virtualscan(SimpleVirtualScan *svs, int beamnum, int size, double minfloor, double step) {
    // svs.resize(beamnum); // svs is qvector<qvector<simple_virtual_scan>>
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i=index;i<beamnum*size;i+=stride)
    {
        int j = i%size;
        // if(index == 0) {
        //     printf("beamnum*size: %d\n", beamnum*size);
        //     printf("stride: %d\n", stride);
        //     printf("index: %d\n", index);
        //     printf("i: %d\n", i);
        // }

        svs[i].rotid=j;
        svs[i].length=MAXVIRTUALSCAN;
        svs[i].rotlength=MAXVIRTUALSCAN;
        svs[i].rotheight=minfloor+(j+0.5)*step;
        svs[i].height=minfloor+(j+0.5)*step;
    }
}
__global__
void sortsKernel(SimpleVirtualScan *svs, SimpleVirtualScan *svsback, int beamnum, int size, double obstacleMinHeight, double maxBackDistance, double c, double s) {
    // svs.resize(beamnum); // svs is qvector<qvector<simple_virtual_scan>>
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i=index;i<beamnum;i+=stride) {
        int j;
        bool flag=1;
        int startid=0;
        for(j=0;j<size;j++)
        {
            if(flag)
            {
                if(svs[i*size + j].rotlength<MAXVIRTUALSCAN)
                {
                    flag=0;
                    startid=j;
                }
                continue;
            }
            if(svs[i*size + j].rotlength<MAXVIRTUALSCAN && startid==j-1)
            {
                startid=j;
            }
            else if(svs[i*size + j].rotlength<MAXVIRTUALSCAN)
            {
                if(svs[i*size + j].height-svs[i*size + startid].height<obstacleMinHeight&&svs[i*size + j].rotlength-svs[i*size + startid].rotlength>-maxBackDistance)
                {
                    double delta=(svs[i*size + j].rotlength-svs[i*size + startid].rotlength)/(j-startid);
                    int k;
                    for(k=startid+1;k<j;k++)
                    {
                        svs[i*size + k].rotlength = svs[i*size + j].rotlength-(j-k)*delta;
                        svs[i*size + k].length = svs[i*size + k].rotlength*c+svs[i*size + k].rotheight*s;
                        svs[i*size + k].height = -svs[i*size + k].rotlength*s+svs[i*size + k].rotheight*c;
                    }
                }
                startid=j;
            }
        }
        svs[(i*size)+size-1].rotlength=MAXVIRTUALSCAN;
        // thrust::copy(thrust::device, svs+(i*size), svs+(i*size)+size, svsback+(i*size));
        // thrust::sort(thrust::device, svs+(i*size),svs+(i*size)+size,compareDistance);
    }
}
__global__
void sortsCopyKernel(SimpleVirtualScan *svs, SimpleVirtualScan *svsback, int beamnum, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i=index;i<beamnum*size;i+=stride) {
        svsback[i] = svs[i];
    }
}
/*__global__
void sortsSortKernel(SimpleVirtualScan *svs, int beamnum, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // printf("stride: %d index: %d", stride, index);

    for(int i=index;i<beamnum;i+=stride) {
        thrust::stable_sort(thrust::device, svs+(i*size),svs+(i*size)+size,compareDistance);
    }
}*/
__global__
void sortsSortKernel(SimpleVirtualScan *svs, int beamnum, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    SimpleVirtualScan cache[SIZE];
     

    // printf("stride: %d index: %d", stride, index);

    for(int i=index;i<beamnum;i+=stride) {
    	for(int j=0; j<size; ++j) 
    		cache[j] = *(svs + i*size + j);
        thrust::sort(thrust::device, cache,cache+size,compareDistance);
    	for(int j=0; j<size; ++j) 
    		*(svs + i*size + j) = cache[j];
    }
}
__global__
void enumerationSort(SimpleVirtualScan *svs, int beamnum, int size) {
    int cnt = 0;
    int tid = threadIdx.x;
    int ttid = blockIdx.x * blockDim.x + tid;
    SimpleVirtualScan val = svs[ttid];
    __shared__ SimpleVirtualScan cache[SIZE];

    for ( int i = tid; i < size; i += blockDim.x ){
        cache[tid] = svs[i];
        __syncthreads();
        for ( int j = 0; j < blockDim.x; ++j ) {
            if ( compareDistance(cache[j], val) )
                cnt++;
        }rtml1234
        
        __syncthreads();
    }
    *(svs + blockIdx.x * blockDim.x + tid) = val;
    //svs_sort[cnt] = val;
}
void FastVirtualScan::calculateVirtualScans(const std::vector<cv::Vec3f> &pointcloud, int beamNum, double heightStep, double minFloor, double maxCeiling, double obstacleMinHeight, double maxBackDistance, double beamRotation, double minRange)
{

    assert(minFloor<maxCeiling);

    beamnum=beamNum; // 720
    step=heightStep;
    minfloor=minFloor;
    maxceiling=maxCeiling;
    rotation=beamRotation;
    minrange=minRange;
    double c=cos(rotation);
    double s=sin(rotation);

    double PI=3.141592654;
    double density=2*PI/beamnum;

    int size=int((maxceiling-minfloor)/step+0.5); // 160


    //initial Simple Virtual Scan
    timers.resetTimer("initial_simple_virtual_scan");
    // svsback.resize(beamnum); // beamnum = 720

    // std::cout << "about to init_simp" << std::endl;
    int blockSize = 256;
    int numBlocks = (beamnum*size + blockSize - 1) / blockSize;
    init_simple_virtualscan<<<numBlocks, blockSize>>>(svs, beamnum, size, minfloor, step);
    cudaDeviceSynchronize();
    // std::cout << "finished init_simp" << std::endl;
    timers.pauseTimer("initial_simple_virtual_scan");




    //set SVS
    timers.resetTimer("set_svs");
    {
        // std::cout << "int set_svs" << std::endl;
        // char * tmpdata=(char *)(velodyne->data.data());
        // int i,n=velodyne->height*velodyne->width;
        int i,n=pointcloud.size();

        //O(P)
        for(i=0;i<n;i++)
        {
            const cv::Vec3f &point = pointcloud[i];
            double length=sqrt(point[0]*point[0]+point[1]*point[1]);
            double rotlength=length*c-point[2]*s;
            double rotheight=length*s+point[2]*c;
            int rotid=int((rotheight-minfloor)/step+0.5);
            if(rotid>=0&&rotid<size)
            {
                double theta=atan2(point[1],point[0]);
                int beamid=int((theta+PI)/density);
                if(beamid<0)
                {
                    beamid=0;
                }
                else if(beamid>=beamnum)
                {
                    beamid=beamnum-1;
                }
                if(length > minrange && svs[beamid*size + rotid].rotlength > rotlength)
                {
                    // std::cout << "beamid*size + rotid: " << beamid*size + rotid << ": " << c+svs[beamid*size + rotid].rotheight*s << std::endl;
                    svs[beamid*size + rotid].rotlength=rotlength;
                    svs[beamid*size + rotid].length=svs[beamid*size + rotid].rotlength*c+svs[beamid*size + rotid].rotheight*s;
                    svs[beamid*size + rotid].height=-svs[beamid*size + rotid].rotlength*s+svs[beamid*size + rotid].rotheight*c;
                }
            }
        }
    }
    timers.pauseTimer("set_svs");
    std::cout << "svs_set done" << std::endl;




    //sorts
    timers.resetTimer("sorts");
    {
// #ifdef USEOMP
//#ifndef QT_DEBUG
//#pragma omp parallel for \
    default(shared) \
    schedule(static)
//#endif
//#endif
        timers.resetTimer("sortsKernel");
        int blockSize = 32;
        int numBlocks = (beamnum + blockSize - 1) / blockSize; // 23 blocks, for blockSize of 32
        sortsKernel<<<numBlocks, blockSize>>>(svs, svsback, beamnum, size, obstacleMinHeight, maxBackDistance, c, s);
        cudaDeviceSynchronize();
        timers.pauseTimer("sortsKernel");

        timers.resetTimer("sortsCopyKernel");
        blockSize = size;
        numBlocks = (beamnum*size + blockSize - 1) / blockSize; // 450 blocks, for blockSize of 256
        sortsCopyKernel<<<numBlocks, blockSize>>>(svs, svsback, beamnum, size);
        cudaDeviceSynchronize();

        // for(int i=0;i<beamnum*size;i++) {
        //     svsback[i] = svs[i];
        // }

        // std::copy(svs, svs+(beamnum*size), svsback);
        //timers.pauseTimer("sortsCopyKernel");

        //timers.resetTimer("sortsSortKernel");
        /*blockSize = 64;
        numBlocks = (beamnum + blockSize - 1) / blockSize; // = 12 blocks
        sortsSortKernel<<<numBlocks, blockSize>>>(svs, beamnum, size);
        cudaDeviceSynchronize();*/
// #ifdef USEOMP
// #ifndef QT_DEBUG
// #pragma omp parallel for \
//     default(shared) \
//     schedule(static)
// #endif
// #endif
        // for(int i=0;i<beamnum;++i) {
        //     thrust::sort(thrust::host, svs+(i*size),svs+(i*size)+size,compareDistanceHost);
        // }
        
        //SimpleVirtualScan *svs_sort;
        //const int num_streams = 1;
        //cudaStream_t streams[num_streams];
        // std::cout << "cudaMallocManaged ret: " << 
        //cudaMallocManaged(&svs_sort, beamnum*size*sizeof(SimpleVirtualScan));
        //  << std::endl;
        /*for(int i=0;i<num_streams;++i) {
            cudaStreamCreate(&streams[i]);
            enumerationSort<<<blockSize,numBlocks, size*sizeof(SimpleVirtualScan), 
                streams[i]>>>(svs+(i*size), svs_sort+(i*size), beamnum, size);
        }
        */
        
        //dim3 grid(90,1);
        //dim3 block(160, 8);
        //printf("calling enumSort\n");
        enumerationSort<<<beamnum, size>>>(svs, beamnum, size);
        cudaDeviceSynchronize();
        //printf("done calling enumSort\n");
        //cudaFree(svs);
        //svs = svs_sort;
        
        timers.pauseTimer("sortsSortKernel");
    }
    timers.pauseTimer("sorts");
    // std::cout << "sorts done" << std::endl;
}

void FastVirtualScan::getVirtualScan(double thetaminheight, double thetamaxheight, double maxFloor, double minCeiling, double passHeight, QVector<double> &virtualScan)
{
    std::cout << "getVScan" << std::endl;
    virtualScan.fill(MAXVIRTUALSCAN,beamnum);
    minheights.fill(minfloor,beamnum);
    maxheights.fill(maxceiling,beamnum);

    QVector<double> rotVirtualScan;
    rotVirtualScan.fill(MAXVIRTUALSCAN,beamnum);

    int size=int((maxceiling-minfloor)/step+0.5);
    double deltaminheight=fabs(step/tan(thetaminheight));
    double deltamaxheight=fabs(step/tan(thetamaxheight));

#ifdef USEOMP
#ifndef QT_DEBUG
#pragma omp parallel for \
    default(shared) \
    schedule(static)
#endif
#endif
    for(int i=0;i<beamnum;i++)
    {
        int candid=0;
        bool roadfilterflag=1;
        bool denoiseflag=1;
        while(candid<size&&svs[i*size+candid].height>minCeiling)
        {
            candid++;
        }
        if(candid>=size||svs[i*size+candid].rotlength==MAXVIRTUALSCAN)
        {
            virtualScan[i]=0;
            minheights[i]=0;
            maxheights[i]=0;
            continue;
        }
        if(svs[i*size+candid].height>maxFloor)
        {
            virtualScan[i]=svs[i*size+candid].length;
            minheights[i]=svs[i*size+candid].height;
            denoiseflag=0;
            roadfilterflag=0;
        }
        int firstcandid=candid;
        for(int j=candid+1;j<size;j++)
        {
            if(svs[i*size+j].rotid<=svs[i*size+candid].rotid)
            {
                continue;
            }
            int startrotid=svs[i*size+candid].rotid;
            int endrotid=svs[i*size+j].rotid;

            if(svs[i*size+j].rotlength==MAXVIRTUALSCAN)
            {
                if(roadfilterflag)
                {
                    virtualScan[i]=MAXVIRTUALSCAN;
                    minheights[i]=0;//svsback[i][startrotid].height;
                    maxheights[i]=0;//svsback[i][startrotid].height;
                }
                else
                {
                    maxheights[i]=svsback[i*size+startrotid].height;
                }
                break;
            }
            else
            {
                if(denoiseflag)
                {
                    if(startrotid+1==endrotid)
                    {
                        if(svs[i*size+j].rotlength-svs[i*size+candid].rotlength>=deltaminheight)
                        {
                            denoiseflag=0;
                            roadfilterflag=1;
                        }
                        else if(svs[i*size+j].height>maxFloor)
                        {
                            virtualScan[i]=svs[i*size+firstcandid].length;
                            minheights[i]=svs[i*size+firstcandid].height;
                            denoiseflag=0;
                            roadfilterflag=0;
                        }
                    }
                    else
                    {
                        if(svs[i*size+j].height-svs[i*size+candid].height<=passHeight)
                        {
                            if(svs[i*size+j].rotlength-svs[i*size+candid].rotlength<=deltaminheight)
                            {
                                virtualScan[i]=svsback[i*size+startrotid].length;
                                minheights[i]=svsback[i*size+startrotid].height;
                                denoiseflag=0;
                                roadfilterflag=0;
                            }
                            else
                            {
                                virtualScan[i]=svs[i*size+j].length;
                                for(int k=startrotid+1;k<endrotid;k++)
                                {
                                    if(virtualScan[i]>svsback[i*size+k].length)
                                    {
                                        virtualScan[i]=svsback[i*size+k].length;
                                    }
                                }
                                minheights[i]=svsback[i*size+startrotid+1].height;
                                denoiseflag=0;
                                roadfilterflag=0;
                            }
                        }
                        else
                        {
                            continue;
                        }
                    }
                }
                else
                {
                    if(roadfilterflag)
                    {
                        if(startrotid+1==endrotid)
                        {
                            if(svs[i*size+j].rotlength-svs[i*size+candid].rotlength<=deltaminheight)
                            {
                                virtualScan[i]=svsback[i*size+startrotid].length;
                                minheights[i]=svsback[i*size+startrotid].height;
                                roadfilterflag=0;
                            }
                        }
                        else
                        {
                            if(svs[i*size+j].height-svs[i*size+candid].height<=passHeight)
                            {
                                if(svs[i*size+j].rotlength-svs[i*size+candid].rotlength<=deltaminheight)
                                {
                                    virtualScan[i]=svsback[i*size+startrotid].length;
                                    minheights[i]=svsback[i*size+startrotid].height;
                                    roadfilterflag=0;
                                }
                                else
                                {
                                    virtualScan[i]=svs[i*size+j].length;
                                    for(int k=startrotid+1;k<endrotid;k++)
                                    {
                                        if(virtualScan[i]>svsback[i*size+k].length)
                                        {
                                            virtualScan[i]=svsback[i*size+k].length;
                                        }
                                    }
                                    minheights[i]=svsback[i*size+startrotid+1].height;
                                    roadfilterflag=0;
                                }
                            }
                            else
                            {
                                continue;
                            }
                        }
                    }
                    else
                    {
                        if(svs[i*size+j].rotlength-svs[i*size+candid].rotlength>deltamaxheight)
                        {
                            maxheights[i]=svsback[i*size+startrotid].height;
                            break;
                        }
                    }
                }
            }
            candid=j;
        }
        if(virtualScan[i]<=0)
        {
            virtualScan[i]=0;
            minheights[i]=0;
            maxheights[i]=0;
        }
    }
    // std::cout << "out of getVscan" << std::endl;
}
