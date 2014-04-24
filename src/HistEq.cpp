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
}

bool HistEq::runHalideCPU(Image input, Image output, const Params& params) {
	return false;
}

bool HistEq::runHalideGPU(Image input, Image output, const Params& params) {
	return false;
}

bool HistEq::setupOpenCL(cl_context_properties context_prop[], const Params& params, const int image_size) {
	char flags[1024];
	sprintf(flags, "-cl-fast-relaxed-math -D HIST_SIZE=%d -D NUM_CHANNELS=%d -Dimage_size=%d", PIXEL_RANGE, NUM_CHANNELS, image_size);

	if (!initCL(context_prop, params, histEq_kernel, flags)) {
		return false;
	}

	/////////////////////////////////////////////////////////////////kernels
	cl_int err;


	//compute partial histogram
	kernels["partial_hist"] = clCreateKernel(m_program, "partial_hist", &err);
	CHECK_ERROR_OCL(err, "creating partial_hist kernel", return false);

	//merge partial histograms
	kernels["hist"] = clCreateKernel(m_program, "merge_hist", &err);
	CHECK_ERROR_OCL(err, "creating merge_hist kernel", return false);

	//compute cdf of brightness histogram
	kernels["hist_cdf"] = clCreateKernel(m_program, "hist_cdf", &err);
	CHECK_ERROR_OCL(err, "creating hist_cdf kernel", return false);

	//perfrom histogram equalisation to the original image
	kernels["hist_eq"] = clCreateKernel(m_program, "histogram_equalisation", &err);
	CHECK_ERROR_OCL(err, "creating histogram_equalisation kernel", return false);



	//get info to set the global_reduc and local_reduc according to the GPU specs
	size_t preferred_wg_size;	//workgroup size should be a multiple of this
	err = clGetKernelWorkGroupInfo (kernels["partial_hist"], m_device,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &preferred_wg_size, NULL);
	CHECK_ERROR_OCL(err, "getting CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE", return false);

	size_t max_cu;	//max compute units
	err = clGetDeviceInfo(m_device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(size_t), &max_cu, NULL);

	const size_t local_reduc = preferred_wg_size;	//workgroup size for reduction kernels
	const size_t global_reduc = ((int)preferred_wg_size)*((int)max_cu);	//global_reduc size for reduction kernels
	const int num_wg_reduc = global_reduc/local_reduc;

	const size_t merge_hist_local = PIXEL_RANGE;
	const size_t merge_hist_global = PIXEL_RANGE*merge_hist_local;

	const size_t local = preferred_wg_size;
	const size_t global = ceil((float)image_size/(float)local) * local;

	local_sizes["reduc"]  = local_reduc;
	global_sizes["reduc"] = global_reduc;

	local_sizes["merge_hist"] = merge_hist_local;
	global_sizes["merge_hist"] = merge_hist_global;

	local_sizes["hist_cdf"] = PIXEL_RANGE;
	global_sizes["hist_cdf"] = PIXEL_RANGE;

	local_sizes["normal"] = local;
	global_sizes["normal"] = global;


	/////////////////////////////////////////////////////////////////allocating memory

	mems["partial_hist"] = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, sizeof(unsigned int)*PIXEL_RANGE*num_wg_reduc, NULL, &err);
	CHECK_ERROR_OCL(err, "creating histogram memory", return false);

	mems["hist"] = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, sizeof(unsigned int)*PIXEL_RANGE, NULL, &err);
	CHECK_ERROR_OCL(err, "creating histogram memory", return false);

	mems["image"] = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, sizeof(float)*image_size*NUM_CHANNELS, NULL, &err);
	CHECK_ERROR_OCL(err, "creating image memory", return false);



	/////////////////////////////////////////////////////////////////setting kernel arguements

	err  = clSetKernelArg(kernels["partial_hist"], 0, sizeof(cl_mem), &mems["image"]);
	err |= clSetKernelArg(kernels["partial_hist"], 1, sizeof(cl_mem), &mems["partial_hist"]);
	CHECK_ERROR_OCL(err, "setting partial_hist arguments", return false);

	err  = clSetKernelArg(kernels["hist"], 0, sizeof(cl_mem), &mems["partial_hist"]);
	err |= clSetKernelArg(kernels["hist"], 1, sizeof(cl_mem), &mems["hist"]);
	err |= clSetKernelArg(kernels["hist"], 2, sizeof(unsigned int)*local_sizes["merge_hist"], NULL);
	err |= clSetKernelArg(kernels["hist"], 3, sizeof(unsigned int), &num_wg_reduc);
	CHECK_ERROR_OCL(err, "setting merge_hist arguments", return false);

	err = clSetKernelArg(kernels["hist_cdf"], 0, sizeof(cl_mem), &mems["hist"]);
	CHECK_ERROR_OCL(err, "setting hist_cdf arguments", return false);

	err  = clSetKernelArg(kernels["hist_eq"], 0, sizeof(cl_mem), &mems["image"]);
	err |= clSetKernelArg(kernels["hist_eq"], 1, sizeof(cl_mem), &mems["hist"]);
	CHECK_ERROR_OCL(err, "setting histogram_equalisation arguments", return false);

	return true;
}

double HistEq::runCLKernels() {
	cl_int err;
	//let it begin
	double start = omp_get_wtime();

	err = clEnqueueNDRangeKernel(m_queue, kernels["partial_hist"], 1, NULL, &global_sizes["reduc"], &local_sizes["reduc"], 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "enqueuing partial_hist kernel", return false);

	err = clEnqueueNDRangeKernel(m_queue, kernels["hist"], 1, NULL, &global_sizes["merge_hist"], &local_sizes["merge_hist"], 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "enqueuing merge_hist kernel", return false);

	err = clEnqueueNDRangeKernel(m_queue, kernels["hist_cdf"], 1, NULL, &global_sizes["hist_cdf"], &local_sizes["hist_cdf"], 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "enqueuing hist_cdf kernel", return false);

	err = clEnqueueNDRangeKernel(m_queue, kernels["hist_eq"], 1, NULL, &global_sizes["normal"], &local_sizes["normal"], 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "enqueuing histogram_equalisation kernel", return false);

	err = clFinish(m_queue);
	CHECK_ERROR_OCL(err, "running kernels", return false);
	return omp_get_wtime() - start;
}

bool HistEq::runOpenCL(int gl_texture) {
	return false;
}

bool HistEq::runOpenCL(Image input, Image output) {

	cl_int err;

	//transfer memory to the device
	err = clEnqueueWriteBuffer(m_queue, mems["image"], CL_TRUE, 0, sizeof(float)*input.width*input.height*NUM_CHANNELS, input.data, 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "writing image memory", return false);

	double runTime = runCLKernels();

	err = clEnqueueReadBuffer(m_queue, mems["image"], CL_TRUE, 0, sizeof(float)*output.width*output.height*NUM_CHANNELS, output.data, 0, NULL, NULL );
	CHECK_ERROR_OCL(err, "reading image memory", return false);

	reportStatus("Finished OpenCL kernel");

	// Verification
	bool passed = verify(input, output);
	reportStatus(
		"Finished in %lf ms (verification %s)",
		runTime*1000, passed ? "passed" : "failed");

	return passed;
}

bool HistEq::cleanupOpenCL() {
	clReleaseMemObject(mems["image"]);
	clReleaseMemObject(mems["hist"]);
	clReleaseMemObject(mems["partial_hist"]);
	clReleaseKernel(kernels["partial_hist"]);
	clReleaseKernel(kernels["hist"]);
	clReleaseKernel(kernels["hist_cdf"]);
	clReleaseKernel(kernels["hist_eq"]);
	releaseCL();
	return true;
}


bool HistEq::runReference(Image input, Image output) {
	// Check for cached result
	if (m_reference.data) {
		memcpy(output.data, m_reference.data, output.width*output.height*4);
		reportStatus("Finished reference (cached)");
		return true;
	}

	const int hist_size = PIXEL_RANGE;
	unsigned int brightness_hist[hist_size] = {0};
	int brightness;
	float red, green, blue;

	reportStatus("Running reference");
	for (int y = 0; y < input.height; y++) {
		for (int x = 0; x < input.width; x++) {
			red   = getPixel(input, x, y, 0);
			green = getPixel(input, x, y, 1);
			blue  = getPixel(input, x, y, 2);
			brightness = std::max(std::max(red, green), blue)*(hist_size-1);
			brightness_hist[brightness] ++;
		}
	}

	for (int i = 1; i < hist_size; i++) {
		brightness_hist[i] += brightness_hist[i-1];
	}

	float3 rgb;
	float3 hsv;
	for (int y = 0; y < input.height; y++) {
		for (int x = 0; x < input.width; x++) {
			rgb.x = getPixel(input, x, y, 0);
			rgb.y = getPixel(input, x, y, 1);
			rgb.z = getPixel(input, x, y, 2);
			hsv = RGBtoHSV(rgb);		//Convert to HSV to get Hue and Saturation

			hsv.z = ((hist_size-1)*(brightness_hist[(int)hsv.z] - brightness_hist[0]))
						/(input.height*input.width - brightness_hist[0]);

			rgb = HSVtoRGB(hsv);	//Convert back to RGB with the modified brightness for V
			setPixel(output, x, y, 0, rgb.x);
			setPixel(output, x, y, 1, rgb.y);
			setPixel(output, x, y, 2, rgb.z);
		}
	}


	reportStatus("Finished reference");


	// Cache result
	m_reference.width = output.width;
	m_reference.height = output.height;
	m_reference.data = new float[output.width*output.height*NUM_CHANNELS];
	memcpy(m_reference.data, output.data, output.width*output.height*NUM_CHANNELS);

	return true;
}
