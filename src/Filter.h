#pragma once

#include <CL/cl.h>
#include <math.h>
#include <stdarg.h>

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
	uint8_t* host;
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
	float exposure;
} Image;

typedef struct {
	int numImages;
	Image* images;
	size_t width, height;
} LDRI;

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

	virtual bool runHalideCPU(LDRI input, Image output, const Params& params) = 0;
	virtual bool runHalideGPU(LDRI input, Image output, const Params& params) = 0;
	virtual bool runOpenCL(LDRI input, Image output, const Params& params) = 0;
	virtual bool runReference(LDRI input, Image output) = 0;

	virtual void setStatusCallback(int (*callback)(const char*, va_list args));

protected:
	const char *m_name;
	Image m_reference;
	int (*m_statusCallback)(const char*, va_list args);
	void reportStatus(const char *format, ...) const;
	virtual bool verify(LDRI input, Image output, int tolerance=1);

	cl_device_id m_device;
	cl_context m_context;
	cl_command_queue m_queue;
	cl_program m_program;
	bool initCL(const Params& params, const char *source, const char *options);
	void releaseCL();
};

// Image utils
void toFloat(Image &input);
uchar* toChar(Image &image);
buffer_t createHalideBuffer(Image &image);
float clamp(float x, float min, float max);
float getPixel(Image &image, int x, int y, int c);
float getPixelLuminance(float3 pixel_val);
void setPixel(Image &image, int x, int y, int c, float value);
Image readJPG(const char* filePath);
void writeJPG(Image &image, const char* filePath);
float3 RGBtoHSV(float3 rgb);
float3 HSVtoRGB(float3 hsv);
float3 RGBtoXYZ(float3 rgb);
float3 XYZtoRGB(float3 xyz);

// Timing utils
double getCurrentTime();
}
