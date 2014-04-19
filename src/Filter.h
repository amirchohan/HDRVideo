#pragma once

#include <cassert>
#include <CL/cl.h>
#include <math.h>
#include <stdarg.h>

#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_opengl.h>
#include <CL/cl_gl.h>
#include <GL/glx.h>


#define METHOD_REFERENCE  (1<<1)
#define METHOD_HALIDE_CPU (1<<2)
#define METHOD_HALIDE_GPU (1<<3)
#define METHOD_OPENCL     (1<<4)

#define PIXEL_RANGE	256	//8-bit
#define NUM_CHANNELS 4	//RGBA

#define CHECK_ERROR_OCL(err, op, action)							\
	if (err != CL_SUCCESS) {										\
		reportStatus("Error during operation '%s' (%d)", op, err);	\
		releaseCL();												\
		action;														\
	}

#ifndef BUFFER_T_DEFINED
#define BUFFER_T_DEFINED
typedef struct buffer_t {
	uint64_t dev;
	float* host;
	int32_t extent[4];
	int32_t stride[4];
	int32_t min[4];
	int32_t elem_size;
	bool host_dirty;
	bool dev_dirty;
} buffer_t;
#endif
extern "C" void halide_dev_sync(void *user_context);
extern "C" void halide_copy_to_dev(void *user_context, buffer_t *buf);
extern "C" void halide_copy_to_host(void *user_context, buffer_t *buf);
extern "C" void halide_release(void *user_context);

namespace hdr
{
typedef unsigned char uchar;

typedef struct {
	float* data;
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
		_Params_() {
			type = CL_DEVICE_TYPE_ALL;
			platformIndex = 0;
			deviceIndex = 0;
		}
	} Params;

public:
	Filter();
	virtual ~Filter();

	virtual void clearReferenceCache();
	virtual const char* getName() const;

	virtual bool runHalideCPU(Image input, Image output, const Params& params) = 0;
	virtual bool runHalideGPU(Image input, Image output, const Params& params) = 0;
	virtual bool runOpenCL(Image input, Image output, const Params& params) = 0;
	virtual bool runReference(Image input, Image output) = 0;
	virtual Image runFilter(Image input, Params params, unsigned int method);

	virtual void setStatusCallback(int (*callback)(const char*, va_list args));

protected:
	const char *m_name;
	Image m_reference;
	int (*m_statusCallback)(const char*, va_list args);
	void reportStatus(const char *format, ...) const;
	virtual bool verify(Image input, Image output, float tolerance=(1/255.f));

	cl_device_id m_device;
	cl_context m_clContext;
	cl_command_queue m_queue;
	cl_program m_program;
	SDL_GLContext m_glContext;
	bool initCL(const Params& params, const char *source, const char *options);
	void releaseCL();
};

// Timing utils
double getCurrentTime();

// Image utils
buffer_t createHalideBuffer(Image &image);

float* channel_mipmap(float* input, int width, int height, int level=1);
Image image_mipmap(Image &input, int level=1);

float clamp(float x, float min, float max);
float getPixelLuminance(float3 pixel_val);

float getValue(float* data, int x, int y, int width, int height);

float getPixel(Image &image, int x, int y, int c);
void setPixel(Image &image, int x, int y, int c, float value);

Image readJPG(const char* filePath);
void writeJPG(Image &image, const char* filePath);

void toFloat(Image &input);

float3 RGBtoHSV(float3 rgb);
float3 HSVtoRGB(float3 hsv);
float3 RGBtoXYZ(float3 rgb);
float3 XYZtoRGB(float3 xyz);

}
