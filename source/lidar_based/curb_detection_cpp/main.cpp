#include "boundary_detection.h"

// Include VelodyneCapture Header
// #include "VelodyneCapture.h"

int main( int argc, char* argv[] ) {
    if (argc > 1) {
        // Boundary_detection *detection = new Boundary_detection(argv[1], 0, 15.0, 1.125);
        Boundary_detection *detection = new Boundary_detection("test2/", 3, 17, 0.5);
        vector<bool> result = detection->run_detection(true);    
    }
    else {
        Boundary_detection *detection = new Boundary_detection("runHeavyRain.pcap", 0, 16.0, 0.5);
        vector<bool> result = detection->run_detection(true);    
    }
    return 0;
}
