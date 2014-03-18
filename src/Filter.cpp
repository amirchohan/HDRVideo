#include <stddef.h>
#include <stdlib.h>
#include <sys/time.h>
#include <iostream>
#include <exception>
#include <stdexcept>

#include "SDL.h"
#include "SDL_image.h"
#include "jpeglib.h"

#include "Filter.h"

namespace hdr
{
Filter::Filter() {
	m_statusCallback = NULL;
	m_context = 0;
	m_queue = 0;
	m_program = 0;
	m_reference.data = NULL;
}

Filter::~Filter() {
	clearReferenceCache();
}

void Filter::clearReferenceCache() {
	if (m_reference.data) {
		delete[] m_reference.data;
		m_reference.data = NULL;
	}
}

const char* Filter::getName() const {
	return m_name;
}

bool Filter::initCL(const Params& params, const char *source, const char *options) {
	// Ensure no existing context
	releaseCL();

	cl_int err;
	cl_uint numPlatforms, numDevices;

	cl_platform_id platform, platforms[params.platformIndex+1];
	err = clGetPlatformIDs(params.platformIndex+1, platforms, &numPlatforms);
	CHECK_ERROR_OCL(err, "getting platforms", return false);
	if (params.platformIndex >= numPlatforms) {
		reportStatus("Platform index %d out of range (%d platforms found)",
			params.platformIndex, numPlatforms);
		return false;
	}
	platform = platforms[params.platformIndex];

	cl_device_id devices[params.deviceIndex+1];
	err = clGetDeviceIDs(platform, params.type, params.deviceIndex+1, devices, &numDevices);
	CHECK_ERROR_OCL(err, "getting devices", return false);
	if (params.deviceIndex >= numDevices) {
		reportStatus("Device index %d out of range (%d devices found)",
			params.deviceIndex, numDevices);
		return false;
	}
	m_device = devices[params.deviceIndex];

	char name[64];
	clGetDeviceInfo(m_device, CL_DEVICE_NAME, 64, name, NULL);
	reportStatus("Using device: %s", name);

	m_context = clCreateContext(NULL, 1, &m_device, NULL, NULL, &err);
	CHECK_ERROR_OCL(err, "creating context", return false);

	m_queue = clCreateCommandQueue(m_context, m_device, 0, &err);
	CHECK_ERROR_OCL(err, "creating command queue", return false);

	m_program = clCreateProgramWithSource(m_context, 1, &source, NULL, &err);
	CHECK_ERROR_OCL(err, "creating program", return false);

	err = clBuildProgram(m_program, 1, &m_device, options, NULL, NULL);
	if (err == CL_BUILD_PROGRAM_FAILURE) {
		size_t sz;
		clGetProgramBuildInfo(
			m_program, m_device, CL_PROGRAM_BUILD_LOG, 0, NULL, &sz);
		char *log = (char*)malloc(++sz);
		clGetProgramBuildInfo(
			m_program, m_device, CL_PROGRAM_BUILD_LOG, sz, log, NULL);
		reportStatus(log);
		free(log);
	}
	CHECK_ERROR_OCL(err, "building program", return false);

	reportStatus("OpenCL context initialised.");
	return true;
}

void Filter::releaseCL() {
	if (m_program) {
		clReleaseProgram(m_program);
		m_program = 0;
	}
	if (m_queue) {
		clReleaseCommandQueue(m_queue);
		m_queue = 0;
	}
	if (m_context) {
		clReleaseContext(m_context);
		m_context = 0;
	}
}

void Filter::reportStatus(const char *format, ...) const {
	if (m_statusCallback) {
		va_list args;
		va_start(args, format);
		m_statusCallback(format, args);
		va_end(args);
	}
}

void Filter::setStatusCallback(int (*callback)(const char*, va_list args)) {
	m_statusCallback = callback;
}

bool Filter::verify(LDRI input, Image output, int tolerance) {
	// Compute reference image
	Image ref = {new float[output.width*output.height*3], output.width, output.height};
	runReference(input, ref);


	// Compare pixels
	int errors = 0;
	const int maxErrors = 16;
	for (int y = 0; y < output.height; y++) {
		for (int x = 0; x < output.width; x++) {
			for (int c = 0; c < 3; c++) {
				int r = getPixel(ref, x, y, c);
				int o = getPixel(output, x, y, c);
				int diff = abs(r - o);

				if (diff > tolerance) {
					// Only report first few errors
					if (errors < maxErrors) {
						reportStatus("Mismatch at (%d,%d,%d): %d vs %d", x, y, c, r, o);
					}
					if (++errors == maxErrors) {
						reportStatus("Supressing further errors");
					}
				}
			}
		}
	}

	free(ref.data);
	return errors == 0;
}

/////////////////
// Image utils //
/////////////////

int clamp(int x, int min, int max) {
	return x < min ? min : x > max ? max : x;
}

float clamp(float x, float min, float max) {
	return x < min ? min : x > max ? max : x;
}

buffer_t createHalideBuffer(Image &image) {
	buffer_t buffer = {0};
	buffer.host = toChar(image);
	buffer.extent[0] = image.width;
	buffer.extent[1] = image.height;
	buffer.extent[2] = 4;
	buffer.stride[0] = 4;
	buffer.stride[1] = image.width*4;
	buffer.stride[2] = 1;
	buffer.elem_size = 1;
	return buffer;
}

float getPixel(Image &image, int x, int y, int c) {
	int _x = clamp(x, 0, image.width-1);
	int _y = clamp(y, 0, image.height-1);
	return image.data[(_x + _y*image.width)*3 + c];
}

void setPixel(Image &image, int x, int y, int c, float value) {
	int _x = clamp(x, 0, image.width-1);
	int _y = clamp(y, 0, image.height-1);
	image.data[(_x + _y*image.width)*3 + c] = clamp(value, 0.f, 1.f);
}

float getPixelLuminance(float3 pixel_val) {
	return pixel_val.x*0.2126
			+ pixel_val.y*0.7152
			+ pixel_val.z*0.0722;
}


float3 RGBtoHSV(float3 rgb) {
	float r = rgb.x;
	float g = rgb.y;
	float b = rgb.z;
	float min, max, delta;
	min = std::min(std::min(r, g), b);
	max = std::max(std::max(r, g), b);

	float3 hsv;

	hsv.z = max;	//Brightness
	delta = max - min;
	if(max != 0) hsv.y = delta/max;//Saturation
	else {	// r = g = b = 0	//Saturation = 0, Value is undefined
		hsv.y = 0;
		hsv.x = -1;
		return hsv;
	}

	//Hue
	if(r == max) 		hsv.x = (g-b)/delta;
	else if(g == max) 	hsv.x = (b-r)/delta + 2;
	else 				hsv.x = (r-g)/delta + 4;
	hsv.x *= 60;				
	if( hsv.x < 0 ) hsv.x += 360;

	return hsv;
}

float3 HSVtoRGB(float3 hsv) {
	int i;
	float h = hsv.x;
	float s = hsv.y;
	float v = hsv.z;
	float f, p, q, t;
	float3 rgb;
	if( s == 0 ) { // achromatic (grey)
		rgb.x = rgb.y = rgb.z = v;
		return rgb;
	}
	h /= 60;			// sector 0 to 5
	i = floor( h );
	f = h - i;			// factorial part of h
	p = v * ( 1 - s );
	q = v * ( 1 - s * f );
	t = v * ( 1 - s * ( 1 - f ) );
	switch( i ) {
		case 0:
			rgb.x = v;
			rgb.y = t;
			rgb.z = p;
			break;
		case 1:
			rgb.x = q;
			rgb.y = v;
			rgb.z = p;
			break;
		case 2:
			rgb.x = p;
			rgb.y = v;
			rgb.z = t;
			break;
		case 3:
			rgb.x = p;
			rgb.y = q;
			rgb.z = v;
			break;
		case 4:
			rgb.x = t;
			rgb.y = p;
			rgb.z = v;
			break;
		default:		// case 5:
			rgb.x = v;
			rgb.y = p;
			rgb.z = q;
			break;
	}
	return rgb;
}


float3 RGBtoXYZ(float3 rgb) {
	float3 xyz;
	xyz.x = rgb.x*0.4124 + rgb.y*0.3576 + rgb.z*0.1805;
	xyz.y = rgb.x*0.2126 + rgb.y*0.7152 + rgb.z*0.0722;
	xyz.z = rgb.x*0.0193 + rgb.y*0.1192 + rgb.z*0.9505;
	return xyz;
}

float3 XYZtoRGB(float3 xyz) {
	float3 rgb;
	rgb.x =   xyz.x*3.240479 - xyz.y*1.53715  - xyz.z*0.498535;
	rgb.y = - xyz.x*0.969256 + xyz.y*1.875991 + xyz.z*0.041556;
	rgb.z =   xyz.x*0.055648 - xyz.y*0.204043 + xyz.z*1.057311;
	return rgb;
}


Image readJPG(const char* filePath) {
	SDL_Surface *input    = IMG_Load(filePath);
	if (!input)   throw std::runtime_error("Problem opening input file");
 
 	uchar* udata = (uchar*) input->pixels;
  	float* data = (float*) calloc(3*(input->w * input->h), sizeof(float));
 	for (int i=0; i < 3*(input->w)*(input->h); i++) {
 		data[i] = ((float)udata[i])/255.f;
 	}

 	Image image = {data, input->w, input->h};

 	free(input);

	return image;
}


void init_buffer(jpeg_compress_struct* cinfo) {}
void term_buffer(jpeg_compress_struct* cinfo) {}
boolean empty_buffer(jpeg_compress_struct* cinfo) {
	return TRUE;
}

void writeJPG(Image &img, const char* filePath) {
	FILE *outfile  = fopen(filePath, "wb");

	if (!outfile) throw std::runtime_error("Problem opening output file");
 
	struct jpeg_compress_struct cinfo;
	struct jpeg_error_mgr       jerr;
 
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_compress(&cinfo);
	jpeg_stdio_dest(&cinfo, outfile);

	cinfo.image_width      = img.width;
	cinfo.image_height     = img.height;
	cinfo.input_components = 3;
	cinfo.in_color_space   = JCS_RGB;

	jpeg_set_defaults(&cinfo);
	/*set the quality [0..100]  */
	jpeg_set_quality (&cinfo, 75, true);
	jpeg_start_compress(&cinfo, true);

	uchar* charImageData = toChar(img);

	JSAMPROW row_pointer;          /* pointer to a single row */
 	while (cinfo.next_scanline < cinfo.image_height) {
		row_pointer = (JSAMPROW) &charImageData[cinfo.next_scanline*cinfo.input_components*img.width];
		jpeg_write_scanlines(&cinfo, &row_pointer, 1);
	} 
	jpeg_finish_compress(&cinfo);
}

float weight(float luminance) {
	if (luminance < 0.5) return luminance*2.0;
	else return (1.0 - luminance)*2.0;
}

uchar* toChar(Image &image) {
	uchar* charImageData = (uchar*) calloc(3*image.width*image.height, sizeof(uchar));
	for (int y = 0; y < image.height; y++) {
		for (int x = 0; x < image.width; x++) {
			for (int i=0; i < 3; i++)
				charImageData[(x + y*image.width)*3 + i] = getPixel(image, x, y, i)*255.f;
		}
	}
	return charImageData;
}

//////////////////
// Timing utils //
//////////////////

double getCurrentTime() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_usec + tv.tv_sec*1e6;
}
}
