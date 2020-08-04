#include "fastvirtualscan/fastvirtualscan.h"
#include "../../timers.h"
#include <omp.h>

#define MAXVIRTUALSCAN 1e6
#define USEOMP

/*
rm 4_openmp_gpu_no_cuda_ds_change; rm -rf build; mkdir build; cd build; cmake ..; make -j8; cd ..; ./4_openmp_gpu_no_cuda_ds_change;
*/

extern Timers timers;

void FastVirtualScan::printsvs() {
	int size = 160;
	for(int i =0; i<beamnum*size; ++i) {
		cout << i << ": [" << i/size << "] [" << i%size << "]: ";
		cout << (svs+i)->rotid << " " << (svs+i)->rotlength << " " << 
		(svs+i)->rotheight << " " << (svs+i)->length << " " << 
		(svs+i)->height << endl;
	}
}

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
    
    svs = new SimpleVirtualScan[beamnum*size];
    svsback = new SimpleVirtualScan[beamnum*size];
}

FastVirtualScan::~FastVirtualScan()
{
	//printf("svs pointer: %p svsbsck pointer: %p\n", svs, svsback);
	delete[] svs;
	delete[] svsback;
}

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

    // std::cout << "beamnum: " << beamnum << "\n";
    // std::cout << "step: " << step << "\n";
    // std::cout << "minfloor: " << minfloor << "\n";
    // std::cout << "maxceiling: " << maxceiling << "\n";
    // std::cout << "rotation: " << rotation << "\n";
    // std::cout << "minrange: " << minrange << "\n";

    double PI=3.141592654;
    double density=2*PI/beamnum;

    int size=int((maxceiling-minfloor)/step+0.5); // 160


    //initial Simple Virtual Scan
    timers.resetTimer("initial_simple_virtual_scan");
    {
    	//std::cout << "in init simp vscan\n";
        //svs.resize(beamnum); // svs is qvector<qvector<simple_virtual_scan>>
        //svsback.resize(beamnum); // beamnum = 720
        int nteams = 0, nthreads = 0;
        //std::cout << "in init simp vscan init svs_ptr\n";
        auto svs_cpy = svs;
        //printf("svs_cpy: %p, svs: %p\n", svs_cpy, svs);

#ifdef USEOMP
#ifndef QT_DEBUG
#pragma omp target teams distribute parallel for map(svs_cpy[0:beamnum*size])
#endif
#endif
        for(int i=0;i<beamnum*size;i++)
        {
        	//std::cout << i<< ", in init simp vscan for\n";
        	int j = i%size;
        	
        	// if(i == 0) {
	        //     nteams = omp_get_num_teams();
	        //     nthreads = omp_get_num_threads();
	        // }
            svs_cpy[i].rotid=j;
            svs_cpy[i].length=MAXVIRTUALSCAN;
            svs_cpy[i].rotlength=MAXVIRTUALSCAN;
            svs_cpy[i].rotheight=minfloor+(j+0.5)*step;
            svs_cpy[i].height=minfloor+(j+0.5)*step;
        }
        
        // std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";
        // std::cout << "nteams: " << nteams << " nthreads: " << nthreads << std:: endl;
        
        //svs = svs_ptr;
        //svsback = svsback_ptr;
        //printf("svs_cpy: %p, svs: %p\n", svs_cpy, svs);
        //std::cout << "in init simp vscan done\n";
    }
    timers.pauseTimer("initial_simple_virtual_scan");




    //set SVS
    timers.resetTimer("set_svs");
    {
    	//std::cout << "in set svs\n";
        // char * tmpdata=(char *)(velodyne->data.data());
        // int i,n=velodyne->height*velodyne->width;
        int i,n=pointcloud.size();

        //O(P)
        for(i=0;i<n;i++)
        {
            // float * point=(float *)(tmpdata+i*velodyne->point_step);
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
		            svs[beamid*size + rotid].rotlength=rotlength;
		            svs[beamid*size + rotid].length=svs[beamid*size + rotid].rotlength*c+svs[beamid*size + rotid].rotheight*s;
		            svs[beamid*size + rotid].height=-svs[beamid*size + rotid].rotlength*s+svs[beamid*size + rotid].rotheight*c;
		        }
		    }
        }
    }
    timers.pauseTimer("set_svs");




    //sorts
    timers.resetTimer("sorts");
    {
    	//std::cout << "in sorts\n";
        auto svs_cpy = svs;
        auto svsback_cpy = svsback;
#ifdef USEOMP
#ifndef QT_DEBUG
// #pragma omp parallel for \
//     default(shared) \
//     schedule(static)
#pragma omp target teams distribute parallel for map(svs_cpy[0:beamnum*size], svsback_cpy[0:beamnum*size])
#endif
#endif
        for(int i=0;i<beamnum;i++) {
		    int j;
		    bool flag=1;
		    int startid=0;
		    for(j=0;j<size;j++)
		    {
		        if(flag)
		        {
		            if(svs_cpy[i*size + j].rotlength<MAXVIRTUALSCAN)
		            {
		                flag=0;
		                startid=j;
		            }
		            continue;
		        }
		        if(svs_cpy[i*size + j].rotlength<MAXVIRTUALSCAN && startid==j-1)
		        {
		            startid=j;
		        }
		        else if(svs_cpy[i*size + j].rotlength<MAXVIRTUALSCAN)
		        {
		            if(svs_cpy[i*size + j].height-svs_cpy[i*size + startid].height<obstacleMinHeight&&svs_cpy[i*size + j].rotlength-svs_cpy[i*size + startid].rotlength>-maxBackDistance)
		            {
		                double delta=(svs_cpy[i*size + j].rotlength-svs_cpy[i*size + startid].rotlength)/(j-startid);
		                int k;
		                for(k=startid+1;k<j;k++)
		                {
		                    svs_cpy[i*size + k].rotlength = svs_cpy[i*size + j].rotlength-(j-k)*delta;
		                    svs_cpy[i*size + k].length = svs_cpy[i*size + k].rotlength*c+svs_cpy[i*size + k].rotheight*s;
		                    svs_cpy[i*size + k].height = -svs_cpy[i*size + k].rotlength*s+svs_cpy[i*size + k].rotheight*c;
		                }
		            }
		            startid=j;
		        }
		    }
		    svs_cpy[(i*size)+size-1].rotlength=MAXVIRTUALSCAN;
		    std::copy(svs_cpy+(i*size), svs_cpy+(i*size)+size, svsback_cpy+(i*size));
		    std::sort(svs_cpy+(i*size),svs_cpy+(i*size)+size,compareDistance);
		}
    }
    /*for(int i=0; i<beamnum*size; i++) {
    	std::cout << i / size << "," << i % size << ": " << svs[i].rotid << ", " << svs[i].length << ", " << svs[i].height << std::endl;
    	}*/
    timers.pauseTimer("sorts");
    //std::cout << "sorts done" << std::endl;
}




























void FastVirtualScan::getVirtualScan(double thetaminheight, double thetamaxheight, double maxFloor, double minCeiling, double passHeight, QVector<double> &virtualScan)
{
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
    std::cout << "out of getVscan" << std::endl;
    //printsvs();
}
