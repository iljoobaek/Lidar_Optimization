#include <csignal>

#include "timers.h"
#include <ctime>
#include <chrono>
#include <stdlib.h>
#include <time.h>

#include "boundary_detection.h"
#include "viewer.h"
#include "sensor_config.h"

#include "fastvirtualscan/fastvirtualscan.h"



# include <iostream>
// # include <Python.h>
// #include <Boost/Python.h>
#include <Python.h>
#include <python2.7/Python.h>
#include <boost/python.hpp>
//#include "/home/rtml/.local/lib/python2.7/site-packages/numpy/core/include/numpy/arrayobject.h"
#include "/home/rtml/Kamal_Workspace/Lidar_curb_detection/source/vscan_based/env/lib/python2.7/site-packages/numpy/core/include/numpy/arrayobject.h"
#include <exception>
#include <fstream>

Timers timers = Timers();
extern std::string optType;
extern std::string optInfo;

// Parameters for virtual scan
static int BEAMNUM = 720;
static double STEP = 0.05;
static double MINFLOOR = -2.0;
static double MAXFLOOR = -1.0;
static double MAXCEILING = 6.0;
static double MINCEILING = -0.5;
static double ROADSLOPMINHEIGHT = 80.0;
static double ROADSLOPMAXHEIGHT = 30.0;
static double ROTATION = 3.0;
static double OBSTACLEMINHEIGHT = 1.0;
static double MAXBACKDISTANCE = 1.0;
static double PASSHEIGHT = 2.0;

static double MAXRANGE = 20.0;
static double MINRANGE = 2.0;
static double GRIDSIZE = 10.0;
static double IMAGESIZE = 1000.0;

static volatile sig_atomic_t sig_caught = 0;

void signalHandler(int signum)
{
    sig_caught = 1;
}

std::vector<std::vector<float>> getVscanResult(const FastVirtualScan &virtualscan, const QVector<double> &beams)
{
    std::vector<std::vector<float>> res;
    double density = 2 * PI / BEAMNUM;
    for (int i = 0; i < BEAMNUM; i++)
    {
        double theta = i * density - PI;
        if (beams[i] == 0 || virtualscan.minheights[i] == virtualscan.maxheights[i])
        {
            continue;
        }
        float x = beams[i] * std::cos(theta);
        float y = beams[i] * std::sin(theta);
        float minHeight = virtualscan.minheights[i];
        float maxHeight = virtualscan.maxheights[i];
        res.push_back({x, y, minHeight, maxHeight});
    }  
    return res; 
}

std::vector<cv::Vec3f> vectorToVec3f(const std::vector<std::vector<float>> &vec)
{
    std::vector<cv::Vec3f> res;
    for (auto &v : vec)
    {
        res.push_back(cv::Vec3f(v[0], v[1], 0.0f));
    }
    return res;
}

std::string getEvalFilename(const std::string &root, int frameIdx)
{
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(10) << frameIdx;
    std::string filename = root + "/gt_" + ss.str() + "_r.txt";
    return filename; 
}

cv::viz::WPolyLine generateWPolyLine(std::vector<float> coeffs, float minY, float maxY)
{
    std::vector<cv::Vec3f> linePoints;
    for (int i = minY * 100; i <= maxY * 100; i++) {
        // Check the order of coeffs !
        linePoints.push_back(cv::Vec3f(coeffs[3] + coeffs[2] * i / 100. + coeffs[1] * powf(i / 100., 2) + coeffs[0] * powf(i / 100., 3), i / 100., 0));
    }
    cv::Mat pointsMat = cv::Mat(static_cast<int>(linePoints.size()), 1, CV_32FC3, &linePoints[0]);
    return cv::viz::WPolyLine(pointsMat, cv::viz::Color::blue());
}

cv::viz::WPolyLine generateGTWPolyLine(const std::string &root, int frameIdx)
{
    std::string filename = getEvalFilename(root, frameIdx);
    std::ifstream f(filename);
    std::vector<std::vector<float>> data;
    if (f.is_open()) 
    {
        std::string line;
        while (getline(f, line))
        {
            std::stringstream ss(line);
            std::vector<float> v;
            float num;   
            while (ss >> num)
            {
                v.push_back(num);   
            }
            data.push_back(v);
        }
        std::vector<float> y;
        for (int i = 1; i < data.size(); i++)
        {
            y.push_back(data[i][1]);   
        }
        auto minmaxY = std::minmax_element(y.begin(), y.end());
        for (auto & i : data[0])
        {
            //std::cout << i << "  ";
        }
        //std::cout << " \n Completed writing the line \n";
        //std::cout << std::endl;
        return generateWPolyLine(data[0], *minmaxY.first, *minmaxY.second);
    }
    else
    {
        std::cerr << "file not opened\n";
    }
    std::vector<cv::Vec3f> zero;
    zero.push_back(cv::Vec3f(0, 0, 0));
    cv::Mat pointsMat = cv::Mat(static_cast<int>(zero.size()), 1, CV_32FC3, &zero[0]);
    return cv::viz::WPolyLine(pointsMat, cv::viz::Color::blue());
}


///////////////////////////////////////////////////////////////////////////////
namespace py = boost::python;

template<typename T>
PyArrayObject* vector_to_nparray(const std::vector< std::vector<T> >& vec){

   int type_num = PyArray_FLOAT;
   // rows not empty
   if( !vec.empty() ){

      // column not empty
      if( !vec[0].empty() ){

        size_t nRows = vec.size();
        size_t nCols = vec[0].size();
        npy_intp dims[2] = {nRows, nCols};
        PyArrayObject* vec_array = (PyArrayObject *) PyArray_SimpleNew(2, dims, type_num);

        T *vec_array_pointer = (T*) PyArray_DATA(vec_array);

        // copy vector line by line ... maybe could be done at one
        for (size_t iRow=0; iRow < vec.size(); ++iRow){

          if( vec[iRow].size() != nCols){
             Py_DECREF(vec_array); // delete
             throw(std::string("Can not convert vector<vector<T>> to np.array, since c++ matrix shape is not uniform."));
          }

          std::copy(vec[iRow].begin(),vec[iRow].end(),vec_array_pointer + iRow*nCols);
        }

        return vec_array;

     // Empty columns
     } else {
        // npy_intp dims[2] = {std::vec.size(), 0};
        // return (PyArrayObject*) PyArray_ZEROS(2, dims, PyArray_FLOAT, 0);
     }


   // no data at all
   } else {
      npy_intp dims[2] = {0, 0};
      return (PyArrayObject*) PyArray_ZEROS(2, dims, PyArray_FLOAT, 0);
   }

}

PyObject* vectorToList_Float(const std::vector<float> &data) {
  PyObject* listObj = PyList_New( data.size() );
	if (!listObj) throw std::logic_error("Unable to allocate memory for Python list");
	for (unsigned int i = 0; i < data.size(); i++) {
		PyObject *num = PyFloat_FromDouble( (double) data[i]);
		if (!num) {
			Py_DECREF(listObj);
			throw std::logic_error("Unable to allocate memory for Python list");
		}
		PyList_SET_ITEM(listObj, i, num);
	}
	return listObj;
}
PyObject* NullListObj(void) {
    //std::cout<<"Reached here \n";
    PyObject* listObj = PyList_New( 1 );
    PyObject *num = PyFloat_FromDouble( 0.0 );
    PyList_SET_ITEM(listObj, 0, num);
    return listObj;
}

std::vector<float> listTupleToVector_Float(PyObject* incoming) {
	std::vector<float> data;
	if (PyTuple_Check(incoming)) {
		for(Py_ssize_t i = 0; i < PyTuple_Size(incoming); i++) {
			PyObject *value = PyTuple_GetItem(incoming, i);
			data.push_back( PyFloat_AsDouble(value) );
		}
	} else {
		if (PyList_Check(incoming)) {
			for(Py_ssize_t i = 0; i < PyList_Size(incoming); i++) {
				PyObject *value = PyList_GetItem(incoming, i);
				data.push_back( PyFloat_AsDouble(value) );
			}
		} else {
			throw std::logic_error("Passed PyObject pointer was not a list or tuple!");
		}
	}
	return data;
}

/////////////////////////////////////////////////////////////////////////////////////////

std::string evalPath = "/home/rtml/Kamal_Workspace/Lidar_curb_detection/source/evaluation/gt_generator/evaluation_result";

int main(int argc, char* argv[]) 
{
    setupTimers(timers);
    // Number of velodyne sensors, maximum 6
    int numOfVelodynes;
    int frameStart, frameEnd;
    std::string processNum;
    if (argc < 2)
    {
        numOfVelodynes = 6;
        frameStart = 601;
        frameEnd = 700;
    } 
    else if (argc >= 3) 
    {
        //numOfVelodynes = std::stoi(argv[1]);
        numOfVelodynes = 6;
        frameStart = std::stoi(argv[1]);
        frameEnd = std::stoi(argv[2]);
        processNum = argv[3];
        if (numOfVelodynes < 1 || numOfVelodynes > 6)
        {
            std::cerr << "Invalid number of Velodynes, should be from 1 to 6.\n";
            return -1;
        }
    }
    else {
        std::cerr << "Invalid arguments.\n";
        return -1;
    }
    numOfVelodynes = 1;

    // Signal Handler for pause/resume viewer
    std::signal(SIGINT, signalHandler);
    double total_ms = 0.0;

    // // Get pcap file names
    // std::vector<std::string> pcap_files = SensorConfig::getPcapFiles(); 
    
    // Create Viz3d Viewer and register callbacks
    //cv::viz::Viz3d viewer( "Velodyne" );
    //bool pause(false);
    //LidarViewer::cvViz3dCallbackSetting(viewer, pause);
    
    // Get rotation and translation parameters from lidar to vehicle coordinate
    auto rot_params = SensorConfig::getRotationParams();
    auto rot_vec = SensorConfig::getRotationMatrices(rot_params);
    auto trans_vec = SensorConfig::getTranslationMatrices();

    // Boundary detection object
    //int frameStart = 601, frameEnd = 650;
    // int frameStart = 777, frameEnd = 787;
    // int frameStart = 300, frameEnd = 400;
    Boundary_detection detection(0., 1.125, "/home/rtml/Lidar_curb_detection/data/", "20191126163620_synced/", frameStart, frameEnd+1);

    int numFrames = 1 + frameEnd - frameStart;

    // Virtual scan object
    FastVirtualScan virtualscan = FastVirtualScan(BEAMNUM, STEP, MINFLOOR, MAXCEILING); // cuda and openmp_gpu
    fusion::FusionController fuser;
    
    Py_Initialize();
    PyObject *pName, *pModule, *pDict, *pFunc, *pArgs, *pValue;
    
    // Import tracker
    pModule = PyImport_Import(PyString_FromString("tracking"));
    pDict = PyModule_GetDict(pModule);
    // get functions
    pFunc = PyDict_GetItemString(pDict, "kalman_filter_chk");
    // define argument
    // define arguement type
    // pValue = PyFloat_FromDouble(1);

    // Location for storing prev lines
    std::vector<float> prev_lcoeff;
    std::vector<float> prev_rcoeff;
    std::vector<float> prev_avail_lcoeff;
    std::vector<float> prev_avail_rcoeff;

    std::vector<cv::Vec3f> zero;
    zero.push_back(cv::Vec3f(0, 0, 1));
    cv::Mat pointsMat = cv::Mat(static_cast<int>(zero.size()), 1, CV_32FC3, &zero[0]);
    cv::viz::WPolyLine prev_lline = cv::viz::WPolyLine(pointsMat, cv::viz::Color::gold());
    cv::viz::WPolyLine prev_rline = cv::viz::WPolyLine(pointsMat, cv::viz::Color::gold());

    // Main loop
    int frame_idx = 0;
    //while (detection.isRun()  && !viewer.wasStopped()) 
    while (detection.isRun())
    {

        timers.resetTimer("frame");
        if (sig_caught)
        {
            //std::cout << "Average time per frame: " << (total_ms / (frame_idx-10)) << " ms\n";
            return -1;
        }
        /*if (pause) 
        {
            viewer.spinOnce();
            continue;
        }*/

        std::vector<std::vector<cv::Vec3f>> buffers(numOfVelodynes); 
        std::vector<std::vector<bool>> results(numOfVelodynes); 
        std::vector<std::vector<int>> results_int(numOfVelodynes); 


        // Read in data
        timers.resetTimer("retrieveData");
        detection.retrieveData();
        timers.pauseTimer("retrieveData");


	    timers.resetTimer("pointcloud_preprocessing");
        detection.pointcloud_preprocessing(rot_vec[0]);
	    timers.pauseTimer("pointcloud_preprocessing");
	    
	    //continue;
        

        timers.resetTimer("get_pointcloud");
        auto &pointcloud = detection.get_pointcloud();
        timers.pauseTimer("get_pointcloud");


        LidarViewer::pushToBuffer(buffers[0], pointcloud);
        results[0] = std::vector<bool>(pointcloud.size(), false);
        
        auto t_start = std::chrono::system_clock::now();


















        timers.resetTimer("virtualscan");

        // Run virtualscan algorithm 
        QVector<double> beams; 

        timers.resetTimer("calculateVirtualScans");
        virtualscan.calculateVirtualScans(buffers[0], BEAMNUM, STEP, MINFLOOR, MAXCEILING, OBSTACLEMINHEIGHT, MAXBACKDISTANCE, 
                                          ROTATION * PI / 180.0, MINRANGE);
        timers.pauseTimer("calculateVirtualScans");

        timers.resetTimer("getVirtualScan");
        virtualscan.getVirtualScan(ROADSLOPMINHEIGHT * PI / 180.0, ROADSLOPMAXHEIGHT * PI / 180.0, MAXFLOOR, MINCEILING, 
                                   PASSHEIGHT, beams);
        timers.pauseTimer("getVirtualScan");

        // continue;

        timers.resetTimer("getVscanResult");
        auto res = getVscanResult(virtualscan, beams);
        timers.pauseTimer("getVscanResult");
        
        timers.pauseTimer("virtualscan");
        // std::cout << "virtualscan is done" << std::endl;

        auto t_end = std::chrono::system_clock::now();




















        timers.resetTimer("runDetection");
        auto buf = detection.runDetection(rot_vec[0], trans_vec[0]);
        timers.pauseTimer("runDetection");





        timers.resetTimer("tracking");
        std::vector<cv::viz::WPolyLine> thirdOrder = detection.getThirdOrderLines(buf[1]);
        std::vector<std::vector<cv::Vec3f>> linePoints= detection.getLeftRightLines(buf[1]);
        results[0] = detection.get_result_bool();
        
        std::vector<cv::Vec3f> pc = buf[1];
        std::vector<float> lcoeff = detection.getLeftBoundaryCoeffs();
        std::vector<float> rcoeff = detection.getRightBoundaryCoeffs();

        prev_lcoeff = lcoeff;
        prev_rcoeff = rcoeff;
        //std::cout<< "HERE " << "\t";
        std::vector<float> v_full;
        std::vector<float> v_left;
        std::vector<float> v_right;
        
        for(size_t i=0; i<linePoints[0].size(); ++i) 
        {
            const cv::Vec3f& c = linePoints[0][i];
            v_left.push_back(c[0]);
            v_left.push_back(c[1]);
            v_left.push_back(c[2]);   
            //std::cout<< v_left.size() << "\t";
        }
        
        for(size_t i=0; i<linePoints[1].size(); ++i) 
        {
            const cv::Vec3f& c = linePoints[1][i];
            v_right.push_back(c[0]);
            v_right.push_back(c[1]);
            v_right.push_back(c[2]);   
            //std::cout<< v_right.size() << "\t";
        }

        if ( lcoeff.empty() || rcoeff.empty() ) {
            //std::cout<< "EMPTY ...." << "\n";
            /// LEFT
            if(lcoeff.empty()){
                //std::cout<< "EMPTY .... LEFT" << "\n";
                if (prev_avail_lcoeff.empty()){
                    //printf("Lcoeff is empty \n");
                    std::vector<cv::Vec3f> zero;
                    zero.push_back(cv::Vec3f(0, 0, 1));
                    cv::Mat pointsMat = cv::Mat(static_cast<int>(zero.size()), 1, CV_32FC3, &zero[0]);
                    cv::viz::WPolyLine justLine = cv::viz::WPolyLine(pointsMat, cv::viz::Color::gold());
                    thirdOrder.push_back(justLine); //left
                    
                }
                else{
                    //std::cout<< "EMPTY .... LEFT PREV" << "\n";
                    //std::cout<<"LEFT SIZE: "<< prev_avail_lcoeff.size()<< "\n";
                    prev_lcoeff = prev_avail_lcoeff;
                    // std::vector<cv::viz::WPolyLine> resLine = detection.getLineFromCoeffs(buf[1], prev_lcoeff, 0); //left
                    // thirdOrder.push_back(resLine[0]); //left
                    thirdOrder.push_back(prev_lline);
                }
            }
            else{
                // Update
                /////////////////////////////////////////
                //std::cout<<"Left update \n";
                pArgs = PyTuple_New(5);
                // PyObject* pointsList = vectorToList_Float(v_full);
                PyObject* pointsLeft = vectorToList_Float(v_left);
                PyObject* pointsRight = vectorToList_Float(v_right);
                // PyObject* pointsList = vectorToList_Float(v_full
                PyObject* lList = vectorToList_Float(lcoeff);
                PyObject* rList = NullListObj();
                //std::cout<<"Frame idx one : \t"<<frameStart+frame_idx<<"\n";
                pValue = PyFloat_FromDouble(frameStart+frame_idx);
                
                // PyTuple_SetItem(pArgs, 0, pointsList);
                PyTuple_SetItem(pArgs, 0, pointsLeft);
                PyTuple_SetItem(pArgs, 1, pointsRight);
                PyTuple_SetItem(pArgs, 2, lList);
                PyTuple_SetItem(pArgs, 3, rList);
                PyTuple_SetItem(pArgs, 4, pValue);
                
                int update_flag = 0;

                try {
                    if(PyCallable_Check(pFunc)){
                        // Update line 
                        //std::cout<<"Able to call \n";
                        PyObject * pResult = PyObject_CallObject(pFunc, pArgs);
                        std::vector<float> res_curve = listTupleToVector_Float(pResult);
                        //std::cout<<"Got points Lorig "<<lcoeff[0]<<"\t"<<lcoeff[1]<<"\t"<<lcoeff[2]<<"\n";
                        //std::cout<<"Got points "<<res_curve[0]<<"\t"<<res_curve[1]<<"\t"<<res_curve[2]<<"\n";
                        std::vector<float> l_curv;
                        l_curv.push_back(res_curve[0]);
                        l_curv.push_back(res_curve[1]);
                        l_curv.push_back(res_curve[2]);
                        l_curv.push_back(res_curve[3]);
                        prev_lcoeff = l_curv;
                        prev_avail_lcoeff = prev_lcoeff;
                        std::vector<cv::viz::WPolyLine> resLine = detection.getLineFromCoeffs(buf[1], l_curv, 0); // left
                        thirdOrder.push_back(resLine[0]);
                        prev_lline = resLine[0]; 
                        update_flag = 1;
                    }
                    else{
                        //std::cout<<"Function not present \n";
                    }
                    
                } catch (const py::error_already_set&) {
                    //std::cout<<"Some error \n";
                    PyErr_Print();
                }

                if(update_flag == 0){
                    //printf("Update 0 \n");
                    if (prev_avail_lcoeff.empty()){
                        //printf("Lcoeff is empty \n");
                        std::vector<cv::Vec3f> zero;
                        zero.push_back(cv::Vec3f(0, 0, 1));
                        cv::Mat pointsMat = cv::Mat(static_cast<int>(zero.size()), 1, CV_32FC3, &zero[0]);
                        cv::viz::WPolyLine justLine = cv::viz::WPolyLine(pointsMat, cv::viz::Color::gold());
                        thirdOrder.push_back(justLine); //left
                        update_flag = 1;
                    }
                    else{
                        prev_lcoeff = prev_avail_lcoeff;
                        // std::vector<cv::viz::WPolyLine> resLine = detection.getLineFromCoeffs(buf[1], prev_lcoeff, 0); //left
                        // thirdOrder.push_back(resLine[0]); //left
                        thirdOrder.push_back(prev_lline);
                    }

                }
                /////////////////////////////////////////////

            }

            /// RIGHT
            if(rcoeff.empty()){
                //std::cout<< "EMPTY .... RIGHT" << "\n";
                if (prev_avail_rcoeff.empty()){
                    //printf("Rcoeff is empty \n");
                    std::vector<cv::Vec3f> zero;
                    zero.push_back(cv::Vec3f(0, 0, 1));
                    cv::Mat pointsMat = cv::Mat(static_cast<int>(zero.size()), 1, CV_32FC3, &zero[0]);
                    cv::viz::WPolyLine justLine = cv::viz::WPolyLine(pointsMat, cv::viz::Color::gold());
                    thirdOrder.push_back(justLine); //right
                }
                else{
                    //std::cout<< "EMPTY .... RIGHT PREV" << "\n";
                    prev_rcoeff = prev_avail_rcoeff;
                    // std::vector<cv::viz::WPolyLine> resLine = detection.getLineFromCoeffs(buf[1], prev_rcoeff, 1); //right
                    // thirdOrder.push_back(resLine[0]); //right
                    thirdOrder.push_back(prev_rline);
                }
            }
            else{
                // Update
                ///////////////////////////////////////// 
                //std::cout<<"RIght update \n";
                pArgs = PyTuple_New(5);
                PyObject* pointsLeft = vectorToList_Float(v_left);
                PyObject* pointsRight = vectorToList_Float(v_right);
                PyObject* lList = NullListObj();
                PyObject* rList = vectorToList_Float(rcoeff);
                //std::cout<<"Frame idx two : \t"<<frameStart+frame_idx<<"\n";
                pValue = PyFloat_FromDouble(frameStart+frame_idx);
                
                // PyTuple_SetItem(pArgs, 0, pointsList);
                // PyTuple_SetItem(pArgs, 1, lList);
                // PyTuple_SetItem(pArgs, 2, rList);
                // PyTuple_SetItem(pArgs, 3, pValue);
                PyTuple_SetItem(pArgs, 0, pointsLeft);
                PyTuple_SetItem(pArgs, 1, pointsRight);
                PyTuple_SetItem(pArgs, 2, lList);
                PyTuple_SetItem(pArgs, 3, rList);
                PyTuple_SetItem(pArgs, 4, pValue);
                
                int update_flag = 0;

                try {
                    if(PyCallable_Check(pFunc)){
                        // Update line 
                        PyObject * pResult = PyObject_CallObject(pFunc, pArgs);
                        std::vector<float> res_curve = listTupleToVector_Float(pResult);
                        //std::cout<<"Got points Rorig "<<rcoeff[0]<<"\t"<<rcoeff[1]<<"\t"<<rcoeff[2]<<"\n";
                        //std::cout<<"Got points "<<res_curve[3]<<"\t"<<res_curve[4]<<"\t"<<res_curve[5]<<"\n";
                        std::vector<float> r_curv;
                        r_curv.push_back(res_curve[4]);
                        r_curv.push_back(res_curve[5]);
                        r_curv.push_back(res_curve[6]);
                        r_curv.push_back(res_curve[7]);
                        prev_rcoeff = r_curv;
                        prev_avail_rcoeff = prev_rcoeff;
                        std::vector<cv::viz::WPolyLine> resLine = detection.getLineFromCoeffs(buf[1], r_curv, 1); // right
                        thirdOrder.push_back(resLine[0]);
                        prev_rline = resLine[0];
                        update_flag = 1;
                    }
                    else{
                        //std::cout<<"Function not present \n";
                    }
                    
                } catch (const py::error_already_set&) {
                    //std::cout<<"Some error \n";
                    PyErr_Print();
                }
                if(update_flag == 0){
                    //printf("Update r0 \n");
                    if (prev_avail_rcoeff.empty()){
                        //printf("Rcoeff is empty \n");
                        std::vector<cv::Vec3f> zero;
                        zero.push_back(cv::Vec3f(0, 0, 1));
                        cv::Mat pointsMat = cv::Mat(static_cast<int>(zero.size()), 1, CV_32FC3, &zero[0]);
                        cv::viz::WPolyLine justLine = cv::viz::WPolyLine(pointsMat, cv::viz::Color::gold());
                        thirdOrder.push_back(justLine); //right
                    }
                    else{
                        prev_rcoeff = prev_avail_rcoeff;
                        // std::vector<cv::viz::WPolyLine> resLine = detection.getLineFromCoeffs(buf[1], prev_rcoeff,1); //right
                        // thirdOrder.push_back(resLine[0]); //right
                        thirdOrder.push_back(prev_rline);
                    }

                }
                /////////////////////////////////////////////

            }
            

        }
        
        else{
            //printf("________________________data found both _______________________________%d\n");

            /////////////////////////////////////////
    
            pArgs = PyTuple_New(5);
            PyObject* pointsLeft = vectorToList_Float(v_left);
            PyObject* pointsRight = vectorToList_Float(v_right);
            PyObject* lList = vectorToList_Float(lcoeff);
            PyObject* rList = vectorToList_Float(rcoeff);
            //std::cout<<"Frame idx last : \t"<<frameStart+frame_idx<<"\n";
            pValue = PyFloat_FromDouble(frameStart+frame_idx);
            
            PyTuple_SetItem(pArgs, 0, pointsLeft);
            PyTuple_SetItem(pArgs, 1, pointsRight);
            PyTuple_SetItem(pArgs, 2, lList);
            PyTuple_SetItem(pArgs, 3, rList);
            PyTuple_SetItem(pArgs, 4, pValue);
            
            
            int update_flag = 0;

            try {
                if(PyCallable_Check(pFunc)){
                    // Update line 
                    PyObject * pResult = PyObject_CallObject(pFunc, pArgs);
                    std::vector<float> res_curve = listTupleToVector_Float(pResult);
                    //std::cout<<"Got points lorig "<<lcoeff[0]<<"\t"<<lcoeff[1]<<"\t"<<lcoeff[2]<<"\n";
                    //std::cout<<"Got points rorig "<<rcoeff[0]<<"\t"<<rcoeff[1]<<"\t"<<rcoeff[2]<<"\n";
                    //std::cout<<"Got points "<<res_curve[0]<<"\t"<<res_curve[1]<<"\t"<<res_curve[2]<<"\n";
                    //std::cout<<"Got points "<<res_curve[4]<<"\t"<<res_curve[5]<<"\t"<<res_curve[6]<<"\n";
                    std::vector<float> l_curv;
                    std::vector<float> r_curv;
                    l_curv.push_back(res_curve[0]);
                    l_curv.push_back(res_curve[1]);
                    l_curv.push_back(res_curve[2]);
                    l_curv.push_back(res_curve[3]);
                    r_curv.push_back(res_curve[4]);
                    r_curv.push_back(res_curve[5]);
                    r_curv.push_back(res_curve[6]);
                    r_curv.push_back(res_curve[7]);
                    prev_lcoeff = l_curv;
                    prev_avail_lcoeff = prev_lcoeff;
                    prev_rcoeff = r_curv;
                    prev_avail_rcoeff = prev_rcoeff;
                    std::vector<cv::viz::WPolyLine> resLinel = detection.getLineFromCoeffs(buf[1], prev_lcoeff, 0);
                    std::vector<cv::viz::WPolyLine> resLiner = detection.getLineFromCoeffs(buf[1], prev_rcoeff, 1);
                    thirdOrder.push_back(resLinel[0]);
                    thirdOrder.push_back(resLiner[0]);
                    prev_lline = resLinel[0];
                    prev_rline = resLiner[0];
                    update_flag = 1;
                }
                else{
                    //std::cout<<"Function not present \n";
                }
                
            } catch (const py::error_already_set&) {
                //std::cout<<"Some error \n";
                PyErr_Print();
            }
            if(update_flag == 0){
                //printf("Update both 0 \n");
                // std::vector<cv::Vec3f> zero;
                // zero.push_back(cv::Vec3f(0, 0, 1));
                // cv::Mat pointsMat = cv::Mat(static_cast<int>(zero.size()), 1, CV_32FC3, &zero[0]);
                // cv::viz::WPolyLine justLine = cv::viz::WPolyLine(pointsMat, cv::viz::Color::gold());
                // thirdOrder.push_back(justLine);
                prev_lcoeff = prev_avail_lcoeff;
                // std::vector<cv::viz::WPolyLine> resLinel = detection.getLineFromCoeffs(buf[1], prev_lcoeff, 0); // left
                // thirdOrder.push_back(resLinel[0]);
                thirdOrder.push_back(prev_lline);
                prev_rcoeff = prev_avail_rcoeff;
                // std::vector<cv::viz::WPolyLine> resLiner = detection.getLineFromCoeffs(buf[1], prev_rcoeff, 1); //right
                // thirdOrder.push_back(resLiner[0]);
                thirdOrder.push_back(prev_rline);

            }
            /////////////////////////////////////////////


            // viewer.spinOnce();
            // continue;

        }

        cv::viz::WPolyLine gtLine = generateGTWPolyLine(evalPath, frameStart+frame_idx);

        timers.pauseTimer("tracking");
        

        //std::cout<<"Displaying "<< thirdOrder.size()<< " lines \n";

        // visualization
        //timers.resetTimer("visualization");
        //LidarViewer::updateViewerFromBuffers(buffers, results, viewer, res, thirdOrder, gtLine);
        //timers.pauseTimer("visualization");

        std::chrono::duration<double, std::milli> fp_ms = t_end - t_start;
        //std::cout << "Frame " << frame_idx++ << ": takes " << fp_ms.count() << " ms for vsan" << std::endl;
        if (frame_idx >= 10)
        {
            total_ms += fp_ms.count();
        }
        timers.pauseTimer("frame");
    } // End of main while loop
    
    // Stop python script
    Py_Finalize();
    
    //viewer.close();
    //std::cout << "Average time per frame: " << (total_ms / (frame_idx-10)) << " ms\n";

	// auto now = std::chrono::system_clock::now();
	// auto t_c = std::chrono::system_clock::to_time_t(now - std::chrono::hours(24));
	// std::stringstream ss;
	// ss << std::put_time(std::localtime(&t_c), "%F_%X");
	// std::string timerLogFileName = "timersLog/" + optType + "_" + ss.str() + "_" + processNum + ".txt";
	// std::cout << timerLogFileName << std::endl;
    // timers.printToFile(timerLogFileName, optType, optInfo);

    timers.print();

    return 0;
}
