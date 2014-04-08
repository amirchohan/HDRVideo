#include <cassert>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdlib.h>
#include <sys/stat.h>
#include <map>
#include <sys/types.h>
#include <dirent.h>

#include "HistEq.h"
#include "GammaAdj.h"
#include "Reinhard.h"

using namespace hdr;
using namespace std;

struct _options_ {
	map<string, Filter*> filters;
	map<string, unsigned int> methods;

	_options_() {
		filters["histEq"] = new HistEq();
		filters["gammaAdj"] = new GammaAdj();
		filters["reinhard"] = new Reinhard();

		methods["reference"] = METHOD_REFERENCE;
		methods["opencl"] = METHOD_OPENCL;

#if ENABLE_HALIDE
		methods["halide_cpu"] = METHOD_HALIDE_CPU;
		methods["halide_gpu"] = METHOD_HALIDE_GPU;
#endif
	}
} Options;


void clinfo();
void printUsage();
int updateStatus(const char *format, va_list args);
void checkError(const char* message, int err);
bool is_dir(const char* path);
bool hasEnding (string const &fullString, string const &ending);

int main(int argc, char *argv[]) {
	Filter *filter = NULL;
	Filter::Params params;
	unsigned int method = 0;
	string image_path;

	// Parse arguments
	for (int i = 1; i < argc; i++) {
		if (!filter && (Options.filters.find(argv[i]) != Options.filters.end())) {		//tonemap filter
			filter = Options.filters[argv[i]];
		}
		else if (!method && Options.methods.find(argv[i]) != Options.methods.end()) {
			method = Options.methods[argv[i]];		//implementation method
		}
		else if (!strcmp(argv[i], "-cldevice")) {	//run on the given device
			++i;
			if (i >= argc) {
				cout << "Platform/device index required with -cldevice." << endl;
				exit(1);
			}
			char *next;
			params.platformIndex = strtoul(argv[i], &next, 10);
			if (strlen(next) == 0 || next[0] != ':') {
				cout << "Invalid platform/device index." << endl;
				exit(1);
			}
			params.deviceIndex = strtoul(++next, &next, 10);
			if (strlen(next) != 0) {
				cout << "Invalid platform/device index." << endl;
				exit(1);
			}
		}
		else if (!strcmp(argv[i], "-image")) {	//apply filter on the given image
			++i;
			if (i >= argc) {
				cout << "Invalid image path with -image." << endl;
				exit(1);
			}
			image_path = argv[i];
		}		
		else if (!strcmp(argv[i], "-clinfo")) {
			clinfo();
			exit(0);
		}
	}
	if (filter == NULL || method == 0) {	//invalid arguments
		printUsage();
		exit(1);
	}

	if (image_path == "") image_path = "../test_images/lena-300x300.jpg";
	Image input = readJPG(image_path.c_str());

	// Run filter
	filter->setStatusCallback(updateStatus);
	Image output = filter->runFilter(input, params, method);

	//Save the file
	int lastindex;
	if (is_dir(image_path.c_str()))	image_path = image_path.substr(0, image_path.find_last_of("/"));
	else image_path = image_path.substr(0, image_path.find_last_of("."));

	string image_name = image_path.substr(image_path.find_last_of("/")+1, 100);
	string output_path = "../output_images/" + image_name + "_";
	output_path = output_path + filter->getName() + ".jpg";

	writeJPG(output, output_path.c_str());

	return 0;
}

void clinfo() {
#define MAX_PLATFORMS 8
#define MAX_DEVICES   8
#define MAX_NAME    256
	cl_uint numPlatforms, numDevices;
	cl_platform_id platforms[MAX_PLATFORMS];
	cl_device_id devices[MAX_DEVICES];
	char name[MAX_NAME];
	cl_int err;

	err = clGetPlatformIDs(MAX_PLATFORMS, platforms, &numPlatforms);
	checkError("Error retrieving platforms\n", err);
	if (numPlatforms == 0) {
		cout << "No platforms found." << endl;
		return;
	}

	for (int p = 0; p < numPlatforms; p++) {
		clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, MAX_NAME, name, NULL);
		cout << endl << "Platform " << p << ": " << name << endl;

		err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL,
						 MAX_DEVICES, devices, &numDevices);
		checkError("Error retrieving devices\n", err);

		if (numDevices == 0) {
			cout << "No devices found." << endl;
			continue;
		}
		for (int d = 0; d < numDevices; d++) {
			clGetDeviceInfo(devices[d], CL_DEVICE_NAME, MAX_NAME, name, NULL);
			cout << "-> Device " << d << ": " << name << endl;
		}
	}
	cout << endl;
}


void printUsage() {
	cout << endl << "Usage: hdr FILTER METHOD [-image PATH] [-cldevice P:D]";
	cout << endl << "       hdr -clinfo" << endl;

	cout << endl << "Where FILTER is one of:" << endl;
	map<string, Filter*>::iterator fItr;
	for (fItr = Options.filters.begin(); fItr != Options.filters.end(); fItr++) {
		cout << "\t" << fItr->first << endl;
	}

	cout << endl << "Where METHOD is one of:" << endl;
	map<string, unsigned int>::iterator mItr;
	for (mItr = Options.methods.begin(); mItr != Options.methods.end(); mItr++) {
		cout << "\t" << mItr->first << endl;
	}

	//cout << endl
	//<< "If STITCHING is used then the image PATH, " << endl
	//<< "should be the directory to the input images."
	//<< endl;

	cout << endl
	<< "If specifying an OpenCL device with -cldevice, " << endl
	<< "P and D correspond to the platform and device " << endl
	<< "indices reported by running -clinfo."
	<< endl;

	cout << endl;
}

int updateStatus(const char *format, va_list args) {
	vprintf(format, args);
	printf("\n");
	return 0;
}

void checkError(const char* message, int err) {
	if (err != CL_SUCCESS) {
		printf("%s %d\n", message, err);
		exit(1);
	}
}

bool is_dir(const char* path) {
	struct stat buf;
	stat(path, &buf);
	return S_ISDIR(buf.st_mode);
}

bool hasEnding (string const &fullString, string const &ending) {
	if (fullString.length() >= ending.length()) {
		return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
	} else {
		return false;
	}
}