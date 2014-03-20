#include <string.h>
#include <cstdio>
#include <algorithm>
#include <omp.h>

#include "Stitching.h"
//#include "opencl/exposure.h"
#if ENABLE_HALIDE
#include "halide/blur_cpu.h"
#include "halide/blur_gpu.h"
#endif

/* 
Combines different exposure images to create HDR, as proposed here:
http://www.ceng.metu.edu.tr/~akyuz/files/hdrgpu.pdf
*/

using namespace hdr;

Stitching::Stitching() : Filter() {
	m_name = "Stitching";
	m_type = STITCH;
}

bool Stitching::runHalideCPU(LDRI input, Image output, const Params& params) {
	return false;
}

bool Stitching::runHalideGPU(LDRI input, Image output, const Params& params) {
	return false;
}

bool Stitching::runOpenCL(LDRI input, Image output, const Params& params) {
	return false;
}


bool Stitching::runReference(LDRI input, Image output) {
	// Check for cached result
	if (m_reference.data) {
		memcpy(output.data, m_reference.data, output.width*output.height*4);
		reportStatus("Finished reference (cached)");
		return true;
	}

	reportStatus("\tRunning reference");

	float* hdr_image = (float*) calloc(4*input.height*input.width, sizeof(float));

	float logAvgLum = 0;
	float3 hdr, ldr;
	float Ywhite = 0.f;

	for (int y = 0; y < input.height; y++) {
		for (int x = 0; x < input.width; x++) {

			float weightedSum = 0;
			hdr.x = hdr.y = hdr.z = 0;
			for (int i=0; i < input.numImages; i++) {
				ldr.x = getPixel(input.images[i], x, y, 0)*255.f;
				ldr.y = getPixel(input.images[i], x, y, 1)*255.f;
				ldr.z = getPixel(input.images[i], x, y, 2)*255.f;

				float luminance = getPixelLuminance(ldr);
				float w = weight(luminance);
				float exposure = input.images[i].exposure;

				hdr.x += (ldr.x/exposure) * w;
				hdr.y += (ldr.y/exposure) * w;
				hdr.z += (ldr.z/exposure) * w;

				weightedSum += w;
			}

			hdr.x = hdr.x/(weightedSum + 0.000001);
			hdr.y = hdr.y/(weightedSum + 0.000001);
			hdr.z = hdr.z/(weightedSum + 0.000001);

			output.data[(x + y*input.width)*4 + 0] = hdr.x;
			output.data[(x + y*input.width)*4 + 1] = hdr.y;
			output.data[(x + y*input.width)*4 + 2] = hdr.z;
			output.data[(x + y*input.width)*4 + 3] = getPixelLuminance(hdr);
		}
	}

	reportStatus("\tFinished reference");

	// Cache result
	m_reference.width = output.width;
	m_reference.height = output.height;
	m_reference.data = new float[output.width*output.height*4];
	memcpy(m_reference.data, output.data, output.width*output.height*4);

	return true;
}
