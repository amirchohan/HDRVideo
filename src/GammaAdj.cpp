#include <string.h>
#include <cstdio>
#include <algorithm>
#include <omp.h>

#include "GammaAdj.h"
//#include "opencl/gammaAdj.h"
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

bool GammaAdj::runHalideCPU(Image input, Image output, const Params& params) {
	return false;
}

bool GammaAdj::runHalideGPU(Image input, Image output, const Params& params) {
	return false;
}

bool GammaAdj::runOpenCL(Image input, Image output, const Params& params) {
	return false;
}

bool GammaAdj::runReference(Image input, Image output) {
	// Check for cached result
	if (m_reference.data) {
		memcpy(output.data, m_reference.data, output.width*output.height*4);
		reportStatus("Finished reference (cached)");
		return true;
	}

	reportStatus("Running reference");

	const int hist_size = PIXEL_RANGE;
	unsigned int brightness_hist[hist_size] = {0};
	int brightness;
	float3 rgb;

	for (int y = 0; y < input.height; y++) {
		for (int x = 0; x < input.width; x++) {
			rgb.x = getPixel(input, x, y, 0);
			rgb.y = getPixel(input, x, y, 1);
			rgb.z = getPixel(input, x, y, 2);
			brightness = std::max(std::max(rgb.x, rgb.y), rgb.z)*hist_size;
			brightness_hist[brightness] ++;
		}
	}
	for (int i = 1; i < hist_size; i++) {
		brightness_hist[i] += brightness_hist[i-1];
	}

	float3 hsv;
	float* gamma_adj = (float*) calloc(input.width*input.height, sizeof(float)); 

	for (int y = 0; y < input.height; y++) {
		for (int x = 0; x < input.width; x++) {
			rgb.x = getPixel(input, x, y, 0);
			rgb.y = getPixel(input, x, y, 1);
			rgb.z = getPixel(input, x, y, 2);
			hsv = RGBtoHSV(rgb);		//Convert to HSV to get Hue and Saturation

			//histogram equalisation
			hsv.z = ((hist_size-1)*(brightness_hist[(int)hsv.z] - brightness_hist[0]))
						/(input.height*input.width - brightness_hist[0]);

			hsv.z = clamp(hsv.z, 0.f, PIXEL_RANGE);
			//gamma adjustment
			hsv.z /= PIXEL_RANGE;
			hsv.z = (1 - pow((1 - pow(hsv.z, 0.25)), 0.5))*PIXEL_RANGE;

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
			rgb.x = getPixel(input, x, y, 0);
			rgb.y = getPixel(input, x, y, 1);
			rgb.z = getPixel(input, x, y, 2);
			hsv = RGBtoHSV(rgb);

			hsv.z = 0;
			for (int j = -1; j <= 1; j++) {
				for (int i = -1; i <= 1; i++) {
					_x = clamp(x+i, 0, input.width-1);
					_y = clamp(y+j, 0, input.height-1);
					hsv.z += gamma_adj[ _x + _y*input.width] * sharpen_mask[i+1][j+1];
				}
			}
			hsv.z = hsv.z/8.f + gamma_adj[x + y*input.width];
			hsv.y /= PIXEL_RANGE;
			hsv.y = pow(hsv.y, 1.0)*PIXEL_RANGE;

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
