#include <string.h>
#include <cstdio>
#include <algorithm>
#include <omp.h>

#include "HistEq.h"
#include "opencl/histEq.h"
#if ENABLE_HALIDE
#include "halide/blur_cpu.h"
#include "halide/blur_gpu.h"
#endif

using namespace hdr;

HistEq::HistEq() : Filter() {
	m_name = "HistEq";
	m_type = TONEMAP;
}

bool HistEq::runHalideCPU(LDRI input, Image output, const Params& params) {
	return false;
}

bool HistEq::runHalideGPU(LDRI input, Image output, const Params& params) {
	return false;
}

bool HistEq::runOpenCL(LDRI input, Image output, const Params& params) {
	const size_t local = 1000;
	const size_t global = ceil((float)input.width*input.height/(float)local) * local;
	const size_t cdf_global = 256;	//global size for the k_cdf kernel
	const size_t cdf_local = 256;	//local size for the k_cdf kernel

	char flags[1024];
	sprintf(flags, "-cl-fast-relaxed-math -Dimage_size=%lu -Dnum_workgroups=%lu\n", input.width*input.height, global/local);

	if (!initCL(params, histEq_kernel, flags)) {
		return false;
	}

	cl_int err;
	cl_kernel k_partial_hist, k_hist, k_cdf, k_mod_bright;
	cl_mem mem_image, mem_partial_hist, mem_hist;

	//memory objects
	mem_image = clCreateBuffer(	m_context, CL_MEM_READ_WRITE, 
		sizeof(float)*input.width*input.height*3, NULL, &err);
	CHECK_ERROR_OCL(err, "creating image memory", return false);

	mem_partial_hist = clCreateBuffer(m_context, CL_MEM_READ_WRITE, 
		sizeof(unsigned int)*256*cdf_global, NULL, &err);
	CHECK_ERROR_OCL(err, "creating histogram memory", return false);

	mem_hist = clCreateBuffer(m_context, CL_MEM_READ_WRITE, 
		sizeof(unsigned int)*256, NULL, &err);
	CHECK_ERROR_OCL(err, "creating histogram memory", return false);

	err = clEnqueueWriteBuffer(	m_queue, mem_image, CL_TRUE, 0, 
		sizeof(float)*input.width*input.height*3, input.images[0].data, 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "writing image memory", return false);


	//kernel to compute histogram of histEq
	k_partial_hist = clCreateKernel(m_program, "brightness_partial_hist", &err);
	CHECK_ERROR_OCL(err, "creating brightness_partial_hist kernel", return false);

	err  = clSetKernelArg(k_partial_hist, 0, sizeof(cl_mem), &mem_image);
	err |= clSetKernelArg(k_partial_hist, 1, sizeof(cl_mem), &mem_partial_hist);
	err |= clSetKernelArg(k_partial_hist, 2, sizeof(cl_mem), &mem_hist);
	CHECK_ERROR_OCL(err, "setting brightness_partial_hist arguments", return false);

	//kernel to compute histogram of histEq
	k_hist = clCreateKernel(m_program, "brightness_hist", &err);
	CHECK_ERROR_OCL(err, "creating brightness_hist kernel", return false);

	err  = clSetKernelArg(k_hist, 0, sizeof(cl_mem), &mem_partial_hist);
	err |= clSetKernelArg(k_hist, 1, sizeof(cl_mem), &mem_hist);
	CHECK_ERROR_OCL(err, "setting brightness_hist arguments", return false);

	//kernel to compute cdf of histEq histogram
	k_cdf = clCreateKernel(m_program, "hist_cdf", &err);
	CHECK_ERROR_OCL(err, "creating hist_cdf kernel", return false);

	err = clSetKernelArg(k_cdf, 0, sizeof(cl_mem), &mem_hist);
	CHECK_ERROR_OCL(err, "setting hist_cdf arguments", return false);

	//kernel to modify histEq value in the original image
	k_mod_bright = clCreateKernel(m_program, "modify_brightness", &err);
	CHECK_ERROR_OCL(err, "creating modify_brightness kernel", return false);

	err  = clSetKernelArg(k_mod_bright, 0, sizeof(cl_mem), &mem_image);
	err |= clSetKernelArg(k_mod_bright, 1, sizeof(cl_mem), &mem_hist);
	CHECK_ERROR_OCL(err, "setting modify_brightness arguments", return false);


	//let it begin
	double start = omp_get_wtime();

	err = clEnqueueNDRangeKernel(m_queue, k_partial_hist, 1, NULL, &global, &local, 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "enqueuing partial_histEq_hist kernel", return false);

	err = clEnqueueNDRangeKernel(m_queue, k_hist, 1, NULL, &cdf_global, &cdf_local, 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "enqueuing histEq_hist kernel", return false);

	err = clEnqueueNDRangeKernel(m_queue, k_cdf, 1, NULL, &cdf_global, &cdf_local, 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "enqueuing hist_cdf kernel", return false);

	err = clEnqueueNDRangeKernel(m_queue, k_mod_bright, 1, NULL, &global, &local, 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "enqueuing modify_histEq kernel", return false);

	err = clFinish(m_queue);
	CHECK_ERROR_OCL(err, "running kernel", return false);
	double runTime = omp_get_wtime() - start;

	err = clEnqueueReadBuffer(m_queue, mem_image,
		CL_TRUE, 0, sizeof(float)*output.width*output.height*3, output.data, 0, NULL, NULL );
	CHECK_ERROR_OCL(err, "reading image memory", return false);

	reportStatus("Finished OpenCL kernel");

	// Verification
	bool passed = verify(input, output);
	reportStatus(
		"Finished in %lf ms (verification %s)",
		runTime*1000, passed ? "passed" : "failed");

	clReleaseMemObject(mem_image);
	clReleaseMemObject(mem_hist);
	clReleaseMemObject(mem_partial_hist);
	clReleaseKernel(k_partial_hist);
	clReleaseKernel(k_hist);
	clReleaseKernel(k_cdf);
	clReleaseKernel(k_mod_bright);
	releaseCL();
	return true;
}

bool HistEq::runReference(LDRI input, Image output) {
	// Check for cached result
	if (m_reference.data) {
		memcpy(output.data, m_reference.data, output.width*output.height*4);
		reportStatus("Finished reference (cached)");
		return true;
	}

	unsigned int brightness_hist[256] = {0};
	int brightness;
	float red, green, blue;

	reportStatus("\tRunning reference");
	for (int y = 0; y < input.height; y++) {
		for (int x = 0; x < input.width; x++) {
			red   = getPixel(input.images[0], x, y, 0)*255.f;
			green = getPixel(input.images[0], x, y, 1)*255.f;
			blue  = getPixel(input.images[0], x, y, 2)*255.f;
			brightness = std::max(std::max(red, green), blue);
			brightness_hist[brightness] ++;
		}
	}


	for (int i = 1; i < 256; i++) {
		brightness_hist[i] += brightness_hist[i-1];
	}

	float3 rgb;
	float3 hsv;
	for (int y = 0; y < input.height; y++) {
		for (int x = 0; x < input.width; x++) {
			rgb.x = getPixel(input.images[0], x, y, 0)*255.f;
			rgb.y = getPixel(input.images[0], x, y, 1)*255.f;
			rgb.z = getPixel(input.images[0], x, y, 2)*255.f;
			hsv = RGBtoHSV(rgb);		//Convert to HSV to get Hue and Saturation

			hsv.z = floor(
				(255*(brightness_hist[(int)hsv.z] - brightness_hist[0]))
				/(input.height*input.width - brightness_hist[0]));

			rgb = HSVtoRGB(hsv);	//Convert back to RGB with the modified brightness for V
			setPixel(output, x, y, 0, rgb.x/255.f);
			setPixel(output, x, y, 1, rgb.y/255.f);
			setPixel(output, x, y, 2, rgb.z/255.f);
		}
	}


	reportStatus("\tFinished reference");


	// Cache result
	m_reference.width = output.width;
	m_reference.height = output.height;
	m_reference.data = new float[output.width*output.height*3];
	memcpy(m_reference.data, output.data, output.width*output.height*3);

	return true;
}
