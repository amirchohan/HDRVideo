#include <string.h>
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <omp.h>
#include <vector>

#include "GradDom.h"
#include "opencl/gradDom.h"
#if ENABLE_HALIDE
#include "halide/blur_cpu.h"
#include "halide/blur_gpu.h"
#endif


using namespace hdr;

GradDom::GradDom() : Filter() {
	m_name = "GradDom";
}

bool GradDom::runHalideCPU(Image input, Image output, const Params& params) {
	return false;
}

bool GradDom::runHalideGPU(Image input, Image output, const Params& params) {
	return false;
}

bool GradDom::runOpenCL(Image input, Image output, const Params& params) {
	return false;
}

float* attenuate_func(float* lum, int width, int height) {
	float beta = 0.85;
	int k = 0;
	int k_width = width;	//width and height of the level k in the pyramid
	int k_height = height;
	float* k_lum = lum;
	float k_av_grad = 0.f;
	float* k_gradient;

	std::vector<float*> pyramid;
	std::vector<float> av_grads;
	std::vector<std::pair<unsigned int, unsigned int> > pyramid_sizes;

	for ( ; k_width >= 32 && k_height >= 32; k_height/=2, k_width/=2, k++) {

		//computing gradient magnitude using central differences at level k
		k_av_grad = 0.f;
		k_gradient = (float*) calloc(k_width*k_height, sizeof(float));
		for (int y = 0; y < k_height; y++) {
			for (int x = 0; x < k_width; x++) {
				int x_west  = clamp(x-1, 0, k_width-1);
				int x_east  = clamp(x+1, 0, k_width-1);
				int y_north = clamp(y-1, 0, k_height-1);
				int y_south = clamp(y+1, 0, k_height-1);

				float x_grad = (k_lum[x_west + y*k_width] - k_lum[x_east + y*k_width])/pow(2.f, k+1);
				float y_grad = (k_lum[x + y_south*k_width] - k_lum[x + y_north*k_width])/pow(2.f, k+1);
				k_gradient[x + y*k_width] = sqrt(pow(x_grad, 2) + pow(y_grad, 2));
				k_av_grad += k_gradient[x + y*k_width];
			}
		}
		pyramid.push_back(k_gradient);
		pyramid_sizes.push_back(std::pair< unsigned int, unsigned int >(k_width, k_height));
		av_grads.push_back(k_av_grad/(k_width*k_height));

		printf("%d, %d, %f\n", k_width, k_height, k_av_grad/(k_width*k_height));

		k_lum = channel_mipmap(k_lum, k_width, k_height);
	}


	//computing attenuation functions
	k_gradient = pyramid.back();
	k_width = pyramid_sizes.back().first;
	k_height = pyramid_sizes.back().second;
	float k_alpha = av_grads.back();
	float* k_atten_func;
	k--;

	//attenuation function for the coarsest level
	k_atten_func = (float*) calloc(k_width*k_height, sizeof(float));
	for (int y = 0; y < k_height; y++) {
		for (int x = 0; x < k_width; x++) {
			k_atten_func[x + y*k_width] = (k_alpha/k_gradient[x + y*k_width])*pow(k_gradient[x + y*k_width]/k_alpha, beta);
		}
	}

	pyramid.pop_back();
	pyramid_sizes.pop_back();
	av_grads.pop_back();

	while (! pyramid.empty()) {
		
		float* k1_atten_func = k_atten_func;
		int k1_width = k_width;
		int k1_height = k_height;

		k_gradient = pyramid.back();
		k_width = pyramid_sizes.back().first;
		k_height = pyramid_sizes.back().second;
		float k_alpha = 0.1f*av_grads.back();
		float k_xy_scale_factor;
		float k_xy_atten_func;
		k--;

		//attenuation function for this level
		k_atten_func = (float*) calloc(k_width*k_height, sizeof(float));
		for (int y = 0; y < k_height; y++) {
			for (int x = 0; x < k_width; x++) {

				if (k_gradient[x + y*k_width] != 0) {

					int k1_x = x/2, k1_y = y/2;		//x and y value of the coarser grid

					//neighbours need to be left or right dependent on where we are
					int n_x = (x & 1) ? 1 : -1;
					int n_y = (y & 1) ? 1 : -1;


					//this stops us from going out of bounds
					if ((k1_x + n_x) < 0) n_x = 0;
					if ((k1_y + n_y) < 0) n_y = 0;
					if ((k1_x + n_x) >= k1_width) n_x = 0;
					if ((k1_y + n_y) >= k1_height) n_y = 0;
					if (k1_x == k1_width) k1_x -= 1;
					if (k1_y == k1_height) k1_y -= 1;

					k_xy_atten_func = 9.0*k1_atten_func[k1_x 		+ k1_y			*k1_width]
									+ 3.0*k1_atten_func[k1_x+n_x 	+ k1_y			*k1_width]
									+ 3.0*k1_atten_func[k1_x 		+ (k1_y+n_y)	*k1_width]
									+ 1.0*k1_atten_func[k1_x+n_x 	+ (k1_y+n_y)	*k1_width];

					k_xy_scale_factor = (k_alpha/k_gradient[x + y*k_width])*pow(k_gradient[x + y*k_width]/k_alpha, beta);
				}
				else k_xy_scale_factor = 0.f;
				k_atten_func[x + y*k_width] = (1.f/16.f)*(k_xy_atten_func)*k_xy_scale_factor;
			}
		}

		pyramid.pop_back();
		pyramid_sizes.pop_back();
		av_grads.pop_back();
	}
	return k_atten_func;
}


float* poissonSolver(float* lum, float* div_grad, int width, int height, float terminationCriterea=0.001) {

	float* prev_dr = (float*) calloc(height*width, sizeof(float));
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			prev_dr[x + y*width] = lum[x+y*width];
		}
	}


	float* new_dr = (float*) calloc(height*width, sizeof(float));

	float diff;
	int converged_pixels = 0;
	while (converged_pixels < width*height) {
		diff = 0;
		converged_pixels = 0;
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int x_west  = clamp(x-1, 0, width-1);
				int x_east  = clamp(x+1, 0, width-1);
				int y_north = clamp(y-1, 0, height-1);
				int y_south = clamp(y+1, 0, height-1);

				float prev = prev_dr[x_west + y*width] + prev_dr[x_east + y*width] + prev_dr[x + y_north*width] + prev_dr[x + y_south*width];
				//printf("%f\n", div_grad[x+y*width]);
				new_dr[x + y*width] = 0.25f*(prev - div_grad[x + y*width]);
				diff = new_dr[x + y*width] - prev_dr[x + y*width];
				diff = (diff >= 0) ? diff : -diff;

				if (diff < terminationCriterea) converged_pixels++;
			}
		}
		//printf("%d, %d\n", converged_pixels, width*height);
		float* swap = prev_dr;
		prev_dr = new_dr;
		new_dr = swap;
	}

	return new_dr;
}


float* apply_constant(float* input_lum, float* output_lum, int width, int height) {
	float av_in_lum = 0;
	float av_out_lum = 0;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			av_in_lum  += input_lum[x+y*width];
			av_out_lum += output_lum[x+y*width];
		}
	}
	av_in_lum  /= width*height;
	av_out_lum /= width*height;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			output_lum[x+y*width] += av_out_lum - av_in_lum;
		}
	}
	return output_lum;
}


bool GradDom::runReference(Image input, Image output) {

	// Check for cached result
	if (m_reference.data) {
		memcpy(output.data, m_reference.data, output.width*output.height*4);
		reportStatus("Finished reference (cached)");
		return true;
	}

	reportStatus("Running reference");

	float sat = 1.f;

	//computing logarithmic luminace of the image
	float* lum = (float*) calloc(input.width * input.height, sizeof(float));	//logarithm luminance
	for (int y = 0; y < input.height; y++) {
		for (int x = 0; x < input.width; x++) {
			float3 org_pixel = {getPixel(input, x, y, 0), getPixel(input, x, y, 1), getPixel(input, x, y, 2)};
			lum[x + y*input.width] = log(getPixelLuminance(org_pixel) + 0.000001);
		}
	}

	//computing the attenuation function which will then be multiplied by luminance gradient to acheive attenuated gradient
	float* att_func = attenuate_func(lum, input.width, input.height);	//o(x,y)

	//luminance gradient in forward direction for x and y
	float* grad_x = (float*) calloc(input.width * input.height, sizeof(float));	//H(x,y)
	float* grad_y = (float*) calloc(input.width * input.height, sizeof(float));	//H(x,y)
	for (int y = 0; y < input.height; y++) {
		for (int x = 0; x < input.width; x++) {
			grad_x[x + y*input.width] = (x < input.width-1 ) ? (lum[x+1 +     y*input.width] - lum[x + y*input.width]) : 0;
			grad_y[x + y*input.width] = (y < input.height-1) ? (lum[x   + (y+1)*input.width] - lum[x + y*input.width]) : 0;
		}
	}

	//attenuated gradient achieved by using the previously computed attenuation function
	float* att_grad_x = (float*) calloc(input.height * input.width, sizeof(float));	//G(x,y)
	float* att_grad_y = (float*) calloc(input.height * input.width, sizeof(float));	//G(x,y)
	for (int y = 0; y < input.height; y++) {
		for (int x = 0; x < input.width; x++) {
			att_grad_x[x + y*input.width] = grad_x[x + y*input.width] * att_func[x + y*input.width];
			att_grad_y[x + y*input.width] = grad_y[x + y*input.width] * att_func[x + y*input.width];
		}
	}

	//divG(x,y)
	float* div_grad = (float*) calloc(input.height * input.width, sizeof(float));
	div_grad[0] = 0;
	for (int x = 1; x < input.width; x++) {
		div_grad[x] = att_grad_x[x] - att_grad_x[x-1];
	}
	for (int y = 1; y < input.height; y++) {
		div_grad[y*input.width] = grad_y[y*input.width] - grad_y[(y-1)*input.width];
		for (int x = 1; x < input.width; x++) {
			div_grad[x + y*input.width] = (att_grad_x[x + y*input.width] - att_grad_x[(x-1) + y*input.width])
										+ (att_grad_y[x + y*input.width] - att_grad_y[x + (y-1)*input.width]);
		}
	}


	float* new_dr = poissonSolver(lum, div_grad, input.width, input.height);

	float* new_lum = apply_constant(lum, new_dr, input.width, input.height);

	float3 rgb;
	for (int y = 0; y < input.height; y++) {
		for (int x = 0; x < input.width; x++) {
			//printf("%f, %f\n", lum[x + y*input.width], new_dr[x+y*input.width]);
			rgb.x = pow(getPixel(input, x, y, 0)/exp(lum[x + y*input.width]), sat)*exp(new_lum[x + y*input.width]);
			rgb.y = pow(getPixel(input, x, y, 1)/exp(lum[x + y*input.width]), sat)*exp(new_lum[x + y*input.width]);
			rgb.z = pow(getPixel(input, x, y, 2)/exp(lum[x + y*input.width]), sat)*exp(new_lum[x + y*input.width]);

			setPixel(output, x, y, 0, rgb.x);
			setPixel(output, x, y, 1, rgb.y);
			setPixel(output, x, y, 2, rgb.z);
		}
	}

	reportStatus("Finished reference");

	// Cache result
	m_reference.width = output.width;
	m_reference.height = output.height;
	m_reference.data = new float[output.width*output.height*4];
	memcpy(m_reference.data, output.data, output.width*output.height*4);

	return true;
}
