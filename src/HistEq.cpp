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

bool HistEq::setupOpenCL(cl_context_properties context_prop[], const Params& params) {
	char flags[1024];
	int hist_size = PIXEL_RANGE+1;

	sprintf(flags, "-cl-fast-relaxed-math -D PIXEL_RANGE=%d -D HIST_SIZE=%d -D NUM_CHANNELS=%d -D width=%d -D height=%d -Dimage_size=%d",
			PIXEL_RANGE, hist_size, NUM_CHANNELS, image_width, image_height, image_width*image_height);

	if (!initCL(context_prop, params, histEq_kernel, flags)) {
		return false;
	}


	/////////////////////////////////////////////////////////////////kernels
	cl_int err;

	kernels["transfer_data"] = clCreateKernel(m_program, "transfer_data", &err);
	CHECK_ERROR_OCL(err, "creating transfer_data kernel", return false);

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

	size_t max_wg_size;
	clGetDeviceInfo(m_device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_wg_size, NULL);

	size_t max_cu;	//max compute units
	err = clGetDeviceInfo(m_device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(size_t), &max_cu, NULL);

	size_t* max_work_items = (size_t*) calloc(3, sizeof(size_t));
	err = clGetDeviceInfo(m_device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t)*3, max_work_items, NULL);

	reportStatus("CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: %lu\nCL_DEVICE_MAX_WORK_GROUP_SIZE: %lu\nCL_DEVICE_MAX_COMPUTE_UNITS: %lu", preferred_wg_size, max_wg_size, max_cu);
	reportStatus("CL_DEVICE_MAX_WORK_ITEM_SIZES: %lu, %lu, %lu", max_work_items[0], max_work_items[1], max_work_items[2]);

	const size_t local_reduc = preferred_wg_size;	//workgroup size for reduction kernels
	const size_t global_reduc = ((int)preferred_wg_size)*((int)max_cu);	//global_reduc size for reduction kernels
	const int num_wg_reduc = global_reduc/local_reduc;

	const size_t merge_hist_local = preferred_wg_size;
	const size_t merge_hist_global = hist_size;

	const size_t local = preferred_wg_size;
	const size_t global = ceil((float)image_width*image_height/(float)local) * local;

	//get info to set the global_reduc and local_reduc according to the GPU specs
	preferred_wg_size;	//workgroup size should be a multiple of this
	err = clGetKernelWorkGroupInfo (kernels["hist_eq"], m_device,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &preferred_wg_size, NULL);
	CHECK_ERROR_OCL(err, "getting CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE", return false);

	reportStatus("%lu", preferred_wg_size);

	size_t* local2Dsize = (size_t*) calloc(2, sizeof(size_t));
	size_t* global2Dsize = (size_t*) calloc(2, sizeof(size_t));
	local2Dsize[0] = preferred_wg_size;
	local2Dsize[1] = preferred_wg_size;
	while (local2Dsize[0]*local2Dsize[1] > max_wg_size) {
		local2Dsize[0] /= 2;
		local2Dsize[1] /= 2;
	}
	local2Dsize[0] = 4;
	local2Dsize[1] = 4;
	global2Dsize[0] = clamp(image_width, 0, max_work_items[0]);
	global2Dsize[1] = clamp(image_height, 0, max_work_items[1]);

	global2Dsize[0] = ceil((float)global2Dsize[0]/(float)local2Dsize[0])*local2Dsize[0];
	global2Dsize[1] = ceil((float)global2Dsize[1]/(float)local2Dsize[1])*local2Dsize[1];

	reportStatus("2D kernel sizes: Local=(%lu, %lu) Global=(%lu, %lu)", local2Dsize[0], local2Dsize[1], global2Dsize[0], global2Dsize[1]);

	oneDlocal_sizes["reduc"]  = local_reduc;
	oneDglobal_sizes["reduc"] = global_reduc;

	oneDlocal_sizes["merge_hist"] = merge_hist_local;
	oneDglobal_sizes["merge_hist"] = merge_hist_global;

	oneDlocal_sizes["hist_cdf"] = local_reduc;
	oneDglobal_sizes["hist_cdf"] = hist_size;

	oneDlocal_sizes["normal"] = local;
	oneDglobal_sizes["normal"] = global;

	local_sizes["transfer_data"] = local2Dsize;
	global_sizes["transfer_data"] = global2Dsize;

	/////////////////////////////////////////////////////////////////allocating memory

	mems["partial_hist"] = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, sizeof(unsigned int)*hist_size*num_wg_reduc, NULL, &err);
	CHECK_ERROR_OCL(err, "creating histogram memory", return false);

	mems["hist"] = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, sizeof(unsigned int)*hist_size, NULL, &err);
	CHECK_ERROR_OCL(err, "creating histogram memory", return false);

	mems["image"] = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, sizeof(float)*image_width*image_height*NUM_CHANNELS, NULL, &err);
	CHECK_ERROR_OCL(err, "creating image memory", return false);

	if (params.opengl) {
		mem_images[0] = clCreateFromGLTexture2D(m_clContext, CL_MEM_READ_ONLY, GL_TEXTURE_2D, 0, in_tex, &err);
		CHECK_ERROR_OCL(err, "creating gl input texture", return false);
		
		mem_images[1] = clCreateFromGLTexture2D(m_clContext, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, out_tex, &err);
		CHECK_ERROR_OCL(err, "creating gl output texture", return false);
	}
	else {
		cl_image_format format;
		format.image_channel_order = CL_RGBA;
		format.image_channel_data_type = CL_UNSIGNED_INT8;
		mem_images[0] = clCreateImage2D(m_clContext, CL_MEM_READ_ONLY, &format, image_width, image_height, 0, NULL, &err);
		CHECK_ERROR_OCL(err, "creating input image memory", return false);
	
		mem_images[1] = clCreateImage2D(m_clContext, CL_MEM_WRITE_ONLY, &format, image_width, image_height, 0, NULL, &err);
		CHECK_ERROR_OCL(err, "creating output image memory", return false);
	}

	/////////////////////////////////////////////////////////////////setting kernel arguements

	err  = clSetKernelArg(kernels["transfer_data"], 0, sizeof(cl_mem), &mem_images[0]);
	err |= clSetKernelArg(kernels["transfer_data"], 1, sizeof(cl_mem), &mems["image"]);
	CHECK_ERROR_OCL(err, "setting transfer_data arguments", return false);

	err  = clSetKernelArg(kernels["partial_hist"], 0, sizeof(cl_mem), &mems["image"]);
	err |= clSetKernelArg(kernels["partial_hist"], 1, sizeof(cl_mem), &mems["partial_hist"]);
	CHECK_ERROR_OCL(err, "setting partial_hist arguments", return false);

	err  = clSetKernelArg(kernels["hist"], 0, sizeof(cl_mem), &mems["partial_hist"]);
	err |= clSetKernelArg(kernels["hist"], 1, sizeof(cl_mem), &mems["hist"]);
	err |= clSetKernelArg(kernels["hist"], 2, sizeof(unsigned int)*oneDlocal_sizes["merge_hist"], NULL);
	err |= clSetKernelArg(kernels["hist"], 3, sizeof(unsigned int), &num_wg_reduc);
	CHECK_ERROR_OCL(err, "setting merge_hist arguments", return false);

	err = clSetKernelArg(kernels["hist_cdf"], 0, sizeof(cl_mem), &mems["hist"]);
	CHECK_ERROR_OCL(err, "setting hist_cdf arguments", return false);

	err  = clSetKernelArg(kernels["hist_eq"], 0, sizeof(cl_mem), &mems["image"]);
	err |= clSetKernelArg(kernels["hist_eq"], 1, sizeof(cl_mem), &mem_images[1]);
	err |= clSetKernelArg(kernels["hist_eq"], 2, sizeof(cl_mem), &mems["hist"]);
	CHECK_ERROR_OCL(err, "setting histogram_equalisation arguments", return false);



	return true;
}

double HistEq::runCLKernels() {
	cl_int err;
	//let it begin
	double start = omp_get_wtime();

	err = clEnqueueNDRangeKernel(m_queue, kernels["transfer_data"], 2, NULL, global_sizes["transfer_data"], local_sizes["transfer_data"], 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "enqueuing transfer_data kernel", return false);

	err = clEnqueueNDRangeKernel(m_queue, kernels["partial_hist"], 1, NULL, &oneDglobal_sizes["reduc"], &oneDlocal_sizes["reduc"], 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "enqueuing partial_hist kernel", return false);

	err = clEnqueueNDRangeKernel(m_queue, kernels["hist"], 1, NULL, &oneDglobal_sizes["merge_hist"], &oneDlocal_sizes["merge_hist"], 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "enqueuing merge_hist kernel", return false);

	err = clEnqueueNDRangeKernel(m_queue, kernels["hist_cdf"], 1, NULL, &oneDglobal_sizes["hist_cdf"], &oneDlocal_sizes["hist_cdf"], 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "enqueuing hist_cdf kernel", return false);

	err = clEnqueueNDRangeKernel(m_queue, kernels["hist_eq"], 2, NULL, global_sizes["transfer_data"], local_sizes["transfer_data"], 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "enqueuing histogram_equalisation kernel", return false);

	err = clFinish(m_queue);
	CHECK_ERROR_OCL(err, "running kernels", return false);

	return omp_get_wtime() - start;
}

bool HistEq::runOpenCL(int input_texid, int output_texid) {
	cl_int err;

	err = clEnqueueAcquireGLObjects(m_queue, 2, &mem_images[0], 0, 0, 0);
	CHECK_ERROR_OCL(err, "acquiring GL objects", return false);

	double runTime = runCLKernels();

	err = clEnqueueReleaseGLObjects(m_queue, 2, &mem_images[0], 0, 0, 0);
	CHECK_ERROR_OCL(err, "releasing GL objects", return false);

	reportStatus("Finished OpenCL kernels in %lf ms", runTime*1000);

	return false;
}

bool HistEq::runOpenCL(Image input, Image output) {
	cl_int err;

 	const size_t origin[] = {0, 0, 0};
 	const size_t region[] = {input.width, input.height, 1};
	err = clEnqueueWriteImage(m_queue, mem_images[0], CL_TRUE, origin, region, sizeof(uchar)*input.width*NUM_CHANNELS, 0, input.data, 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "writing image memory", return false);

	double runTime = runCLKernels();

	err = clEnqueueReadImage(m_queue, mem_images[1], CL_TRUE, origin, region, sizeof(uchar)*input.width*NUM_CHANNELS, 0, output.data, 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "reading image memory", return false);

	reportStatus("Finished OpenCL kernel");

	// Verification
	bool passed = verify(input, output);
	reportStatus(
		"Finished in %lf ms (verification %s)",
		runTime*1000, passed ? "passed" : "failed");

	return true;
}

bool HistEq::cleanupOpenCL() {
	clReleaseMemObject(mems["input_image"]);
	clReleaseMemObject(mems["output_image"]);
	clReleaseMemObject(mems["image"]);
	clReleaseMemObject(mems["hist"]);
	clReleaseMemObject(mems["partial_hist"]);
	clReleaseKernel(kernels["transfer_data"]);
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
		memcpy(output.data, m_reference.data, output.width*output.height*NUM_CHANNELS);
		reportStatus("Finished reference (cached)");
		return true;
	}

	const int hist_size = PIXEL_RANGE+1;
	unsigned int brightness_hist[hist_size] = {0};
	int brightness;
	float red, green, blue;

	reportStatus("Running reference");
	for (int y = 0; y < input.height; y++) {
		for (int x = 0; x < input.width; x++) {
			red   = getPixel(input, x, y, 0);
			green = getPixel(input, x, y, 1);
			blue  = getPixel(input, x, y, 2);
			brightness = std::max(std::max(red, green), blue);
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
	m_reference.data = new uchar[output.width*output.height*NUM_CHANNELS];
	memcpy(m_reference.data, output.data, output.width*output.height*NUM_CHANNELS);

	return true;
}
