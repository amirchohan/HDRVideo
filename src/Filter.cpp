#include <stddef.h>
#include <stdlib.h>
#include <sys/time.h>
#include <iostream>
#include <exception>
#include <stdexcept>

#include "jpeglib.h"

#include "Filter.h"

namespace hdr
{
Filter::Filter() {
	m_statusCallback = NULL;
	m_clContext = 0;
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

	// Initialize SDL2
	/*SDL_Init(SDL_INIT_VIDEO);
	// Window mode MUST include SDL_WINDOW_OPENGL for use with OpenGL.
	SDL_Window *window = SDL_CreateWindow( "SDL2/OpenGL Demo", 0, 0, 640, 480, SDL_WINDOW_OPENGL|SDL_WINDOW_RESIZABLE);
	// Create an OpenGL context associated with the window.
	m_glContext = SDL_GL_CreateContext(window);

	// Create CL context properties, add GLX context & handle to DC 
	cl_context_properties properties[] = { 
		CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(), // GLX Context 
		CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(), // GLX Display
		CL_CONTEXT_PLATFORM, (cl_context_properties)platform, // OpenCL platform
		0 
	};*/

	m_clContext = clCreateContext(NULL, 1, &m_device, NULL, NULL, &err);
	CHECK_ERROR_OCL(err, "creating context", return false);

	m_queue = clCreateCommandQueue(m_clContext, m_device, 0, &err);
	CHECK_ERROR_OCL(err, "creating command queue", return false);

	m_program = clCreateProgramWithSource(m_clContext, 1, &source, NULL, &err);
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
	if (m_clContext) {
		clReleaseContext(m_clContext);
		m_clContext = 0;
	}
	if (m_glContext) {
		SDL_GL_DeleteContext(m_glContext);
		m_glContext = 0;
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

bool Filter::verify(Image input, Image output, float tolerance) {
	// Compute reference image8
	Image ref = {(float*) calloc(output.width*output.height*NUM_CHANNELS, sizeof(float)), output.width, output.height};
	runReference(input, ref);

	// Compare pixels
	int errors = 0;
	const int maxErrors = 16;
	for (int y = 0; y < output.height; y++) {
		for (int x = 0; x < output.width; x++) {
			for (int c = 0; c < NUM_CHANNELS; c++) {
				float r = getPixel(ref, x, y, c);
				float o = getPixel(output, x, y, c);
				float diff = abs(r - o);

				if (diff > tolerance) {
					// Only report first few errors
					if (errors < maxErrors) {
						reportStatus("Mismatch at (%d,%d,%d): %f vs %f", x, y, c, r, o);
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

Image Filter::runFilter(Image input, Params params, unsigned int method) {
	Image output = {(float*) calloc(input.width*input.height*4, sizeof(float)), input.width, input.height};

	std::cout << "--------------------------------Tonemapping using " << m_name << std::endl;

	switch (method)
	{
		case METHOD_REFERENCE:
			runReference(input, output);
			break;
		case METHOD_HALIDE_CPU:
			runHalideCPU(input, output, params);
			break;
		case METHOD_HALIDE_GPU:
			runHalideGPU(input, output, params);
			break;
		case METHOD_OPENCL:
			runOpenCL(input, output, params);
			break;
		default:
			assert(false && "Invalid method.");
	}
	return output;
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
	buffer.host = image.data;
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
	return image.data[(_x + _y*image.width)*NUM_CHANNELS + c];
}

void setPixel(Image &image, int x, int y, int c, float value) {
	int _x = clamp(x, 0, image.width-1);
	int _y = clamp(y, 0, image.height-1);
	image.data[(_x + _y*image.width)*NUM_CHANNELS + c] = clamp(value, 0.f, 1.f);
}

float getPixelLuminance(float3 pixel_val) {
	return    pixel_val.x*0.2126
			+ pixel_val.y*0.7152
			+ pixel_val.z*0.0722;
}


float3 RGBtoHSV(float3 rgb) {
	float r = rgb.x*PIXEL_RANGE;
	float g = rgb.y*PIXEL_RANGE;
	float b = rgb.z*PIXEL_RANGE;
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
		rgb.x = rgb.y = rgb.z = v/PIXEL_RANGE;
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
	rgb.x = rgb.x/PIXEL_RANGE;
	rgb.y = rgb.y/PIXEL_RANGE;
	rgb.z = rgb.z/PIXEL_RANGE;
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
	SDL_Surface *input = IMG_Load(filePath);
	if (!input) throw std::runtime_error("Problem opening input file");
 
 	uchar* udata = (uchar*) input->pixels;
  	float* data = (float*) calloc(4*(input->w * input->h), sizeof(float));

	for (int y = 0; y < input->h; y++) {
		for (int x = 0; x < input->w; x++) {
			for (int j=0; j<3; j++) {
 		 		data[(x + y*input->w)*4 + j] = ((float)udata[(x + y*input->w)*3 + j])/255.f;
 		 	}
 		 	data[(x + y*input->w)*4 + 3] = 0.f;
 		 }
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

	uchar* charImageData = (uchar*) calloc(3*img.width*img.height, sizeof(uchar));
	for (int y = 0; y < img.height; y++) {
		for (int x = 0; x < img.width; x++) {
			for (int i=0; i < 3; i++)
				charImageData[(x + y*img.width)*3 + i] = getPixel(img, x, y, i)*255.f;
		}
	}

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


//////////////////
// Timing utils //
//////////////////

double getCurrentTime() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_usec + tv.tv_sec*1e6;
}
}
