#include <string.h>
#include <cstdio>
#include <algorithm>
#include <omp.h>

#include "GammaAdj.h"
#include "opencl/gammaAdj.h"
#if ENABLE_HALIDE
#include "halide/blur_cpu.h"
#include "halide/blur_gpu.h"
#endif

/* 
Tone-mapping algorithm for HDR images as proposed here:
http://mathematica.stackexchange.com/questions/9342/programmatic-approach-to-hdr-photography-with-mathematica-image-processing-funct
*/

using namespace hdr;

GammaAdj::GammaAdj() : Filter() {
	m_name = "GammaAdj";
}

bool GammaAdj::runHalideCPU(LDRI input, Image output, const Params& params) {
	return false;
}

bool GammaAdj::runHalideGPU(LDRI input, Image output, const Params& params) {
	return false;
}

bool GammaAdj::runOpenCL(LDRI input, Image output, const Params& params) {
	return false;
}

bool GammaAdj::runReference(LDRI input, Image output) {
	// Check for cached result
	if (m_reference.data) {
		memcpy(output.data, m_reference.data, output.width*output.height*4);
		reportStatus("Finished reference (cached)");
		return true;
	}

	reportStatus("Running reference");

	float* gamma_adj = (float*) calloc(input.width*input.height, sizeof(float)); 

	//apply double-sided gamma adjustment to the gammaAdj channel
	float3 rgb, hsv;
	for (int y = 0; y < input.height; y++) {
		for (int x = 0; x < input.width; x++) {
			rgb.x = getPixel(input.images[0], x, y, 0);
			rgb.y = getPixel(input.images[0], x, y, 1);
			rgb.z = getPixel(input.images[0], x, y, 2);
			hsv = RGBtoHSV(rgb);

			hsv.z /= 255.f;
			hsv.z = (1 - pow((1 - pow(hsv.z, 0.25)), 0.5))*255.f;

			gamma_adj[x + y*input.width] = hsv.z;
		}
	}

	const float sharpen_mask[3][3] = {
		{-1, -1, -1},
		{-1,  8, -1},
		{-1, -1, -1}
	};


	int _x, _y;
	for (int y = 0; y < input.height; y++) {
		for (int x = 0; x < input.width; x++) {
			rgb.x = getPixel(input.images[0], x, y, 0);
			rgb.y = getPixel(input.images[0], x, y, 1);
			rgb.z = getPixel(input.images[0], x, y, 2);
			hsv = RGBtoHSV(rgb);
			hsv.z = 0;

			for (int j = -1; j <= 1; j++) {
				for (int i = -1; i <= 1; i++) {
					_x = clamp(x+i, 0, input.width-1);
					_y = clamp(y+j, 0, input.height-1);
					hsv.z += gamma_adj[ _x + _y*input.width] * sharpen_mask[i+1][j+1];
				}
			}
			hsv.z = hsv.z/8 + gamma_adj[x + y*input.width];
			hsv.y /= 255.f;
			hsv.y = pow(hsv.y, 0.95)*255.f;
			rgb = HSVtoRGB(hsv);
			setPixel(output, x, y, 0, rgb.x);
			setPixel(output, x, y, 1, rgb.y);
			setPixel(output, x, y, 2, rgb.z);			
		}
	}

	reportStatus("Finished reference");

	// Cache result
	m_reference.width = output.width;
	m_reference.height = output.height;
	m_reference.data = new float[output.width*output.height*3];
	memcpy(m_reference.data, output.data, output.width*output.height*3);

	return true;
}
