#pragma once

#include <map>
#include <math.h>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <CL/cl.h>
#include <CL/cl_gl.h>

#ifdef __ANDROID_API__
	#include <GLES/gl.h>
#else
	#include <GL/gl.h>
#endif

#define METHOD_REFERENCE  (1<<1)
#define METHOD_HALIDE_CPU (1<<2)
#define METHOD_HALIDE_GPU (1<<3)
#define METHOD_OPENCL     (1<<4)

#define PIXEL_RANGE	255	//8-bit
#define NUM_CHANNELS 4	//RGBA

#define CHECK_ERROR_OCL(err, op, action)							\
	if (err != CL_SUCCESS) {										\
		reportStatus("Error during operation '%s' (%d)", op, err);	\
		releaseCL();												\
		action;														\
	}

namespace hdr
{
typedef unsigned char uchar;

typedef struct {
	uchar* data;
	size_t width, height;
} Image;

typedef struct {
	float x;
	float y;
	float z;
} float3;

class Filter {
public:
	typedef struct _Params_ {
		cl_device_type type;
		cl_uint platformIndex, deviceIndex;
		bool opengl, verify;
		_Params_() {
			type = CL_DEVICE_TYPE_ALL;
			opengl = false;
			platformIndex = 0;
			deviceIndex = 0;
			verify = false;
		}
	} Params;

public:
	Filter();
	virtual ~Filter();

	virtual void clearReferenceCache();
	virtual const char* getName() const;

	virtual bool setupOpenCL(cl_context_properties context_prop[], const Params& params) = 0;
	virtual double runCLKernels(bool recomputeMapping) = 0;
	virtual bool runOpenCL(int input_texid, int output_texid, bool recomputeMapping=true) = 0;
	virtual bool runOpenCL(Image input, Image output, bool recomputeMapping=true) = 0;
	virtual bool cleanupOpenCL() = 0;
	virtual bool runReference(Image input, Image output) = 0;
	virtual Image runFilter(Image input, Params params, unsigned int method);
	virtual bool kernel1DSizes(const char* kernel_name);
	virtual bool kernel2DSizes(const char* kernel_name);
	virtual void setImageSize(int width, int height);
	virtual void setImageTextures(GLuint input_texture, GLuint output_texture);

	virtual void setStatusCallback(int (*callback)(const char*, va_list args));

protected:
	const char *m_name;
	Image m_reference;
	int (*m_statusCallback)(const char*, va_list args);
	void reportStatus(const char *format, ...) const;
	virtual bool verify(Image input, Image output, float tolerance=4.f);

	cl_device_id m_device;
	cl_context m_clContext;
	cl_command_queue m_queue;
	cl_program m_program;
	cl_mem mem_images[2];

	size_t max_cu;	//max compute units

	std::map<std::string, cl_mem> mems;
	std::map<std::string, cl_kernel> kernels;
	std::map<std::string, size_t> oneDlocal_sizes;
	std::map<std::string, size_t> oneDglobal_sizes;
	std::map<std::string, size_t*> local_sizes;
	std::map<std::string, size_t*> global_sizes;



	int image_width;
	int image_height;
	GLuint in_tex;
	GLuint out_tex;

	bool initCL(cl_context_properties context_prop[], const Params& params, const char *source, const char *options);
	void releaseCL();
};

// Timing utils
double getCurrentTime();

// Image utils
float* channel_mipmap(float* input, int width, int height, int level=1);
Image image_mipmap(Image &input, int level=1);

float clamp(float x, float min, float max);
float getPixelLuminance(float3 pixel_val);

float getValue(float* data, int x, int y, int width, int height);

float getPixel(Image &image, int x, int y, int c);
void setPixel(Image &image, int x, int y, int c, float value);

void toFloat(Image &input);

float3 RGBtoHSV(float3 rgb);
float3 HSVtoRGB(float3 hsv);
float3 RGBtoXYZ(float3 rgb);
float3 XYZtoRGB(float3 xyz);

}
