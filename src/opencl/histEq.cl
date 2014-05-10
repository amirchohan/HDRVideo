

uchar glVal_to_cl(uint val);
float3 RGBtoHSV(uint4 rgb);
uint4 HSVtoRGB(float3 hsv);

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

kernel void transfer_data(__read_only image2d_t input_image, __global float* image) {
	int2 pos;
	uint4 pixel;
	for (pos.y = get_global_id(1); pos.y < height; pos.y += get_global_size(1)) {
		for (pos.x = get_global_id(0); pos.x < width; pos.x += get_global_size(0)) {
			pixel = read_imageui(input_image, sampler, pos);
			image[(pos.x + pos.y*width)*NUM_CHANNELS + 0] = (float) glVal_to_cl(pixel.x);
			image[(pos.x + pos.y*width)*NUM_CHANNELS + 1] = (float) glVal_to_cl(pixel.y);
			image[(pos.x + pos.y*width)*NUM_CHANNELS + 2] = (float) glVal_to_cl(pixel.z);		
			image[(pos.x + pos.y*width)*NUM_CHANNELS + 3] = (float) glVal_to_cl(pixel.w);
		}
	}
}


//computes the histogram for brightness
kernel void partial_hist(__global float* image, __global uint* partial_histogram) {
	const int global_size = get_global_size(0);
	const int group_size = get_local_size(0);
	const int group_id = get_group_id(0);
	const int lid = get_local_id(0);

	__local uint l_hist[HIST_SIZE];
	for (int i = lid; i < HIST_SIZE; i+=group_size) {
		l_hist[i] = 0;
	}

	int brightness;
	for (int i = get_global_id(0); i < image_size; i += global_size) {
		brightness = max(max(image[i*NUM_CHANNELS + 0], image[i*NUM_CHANNELS + 1]), image[i*NUM_CHANNELS + 2]);
		barrier(CLK_LOCAL_MEM_FENCE);
		atomic_inc(&l_hist[brightness]);
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	for (int i = lid; i < HIST_SIZE; i+=group_size) {
		partial_histogram[i + group_id * HIST_SIZE] = l_hist[i];
	}
}

//requires global work group size to be equal to HIST_SIZE
kernel void merge_hist(__global uint* partial_histogram, __global uint* histogram, __local uint* l_Data, const int num_hists) {
	const int gid = get_global_id(0);

	uint sum = 0;
	for(uint i = 0; i < num_hists; i++)
		sum += partial_histogram[gid + i*HIST_SIZE];

	histogram[gid] = sum;
}

//TODO: even though this takes barely anytime at all, could look into parrallel scan in future
//computes the cdf of the brightness histogram
kernel void hist_cdf( __global uint* hist) {
	const int gid = get_global_id(0);
	const int global_size = get_global_size(0);

	if (gid==0)
		for (int i=1; i<global_size; i++) {
			hist[i] += hist[i-1];
		}
}

//kernel to perform histogram equalisation using the modified brightness cdf
kernel void histogram_equalisation(read_only image2d_t input_image, write_only image2d_t output_image, __global uint* brightness_cdf) {
	int2 pos;
	uint4 pixel;
	for (pos.y = get_global_id(1); pos.y < height; pos.y += get_global_size(1)) {
		for (pos.x = get_global_id(0); pos.x < width; pos.x += get_global_size(0)) {
			pixel = read_imageui(input_image, sampler, pos);

			pixel.x = (uint) glVal_to_cl(pixel.x);
			pixel.y = (uint) glVal_to_cl(pixel.y);
			pixel.z = (uint) glVal_to_cl(pixel.z);
			pixel.w = (uint) glVal_to_cl(pixel.w);

			write_imageui(output_image, pos, pixel);
		}
	}
}

float3 RGBtoHSV(uint4 rgb) {
	float r = rgb.x;
	float g = rgb.y;
	float b = rgb.z;
	float rgb_min, rgb_max, delta;
	rgb_min = min(min(r, g), b);
	rgb_max = max(max(r, g), b);

	float3 hsv;

	hsv.z = rgb_max;	//Brightness
	delta = rgb_max - rgb_min;
	if(rgb_max != 0) hsv.y = delta/rgb_max;//Saturation
	else {	// r = g = b = 0	//Saturation = 0, Value is undefined
		hsv.y = 0;
		hsv.x = -1;
		return hsv;
	}

	//Hue
	if(r == rgb_max) 		hsv.x = (g-b)/delta;
	else if(g == rgb_max) 	hsv.x = (b-r)/delta + 2;
	else 			 		hsv.x = (r-g)/delta + 4;
	hsv.x *= 60;				
	if( hsv.x < 0 ) hsv.x += 360;

	return hsv;
}

uint4 HSVtoRGB(float3 hsv) {
	int i;
	float h = hsv.x;
	float s = hsv.y;
	float v = hsv.z;
	float f, p, q, t;
	uint4 rgb;
	rgb.w = 0;
	if( s == 0 ) { // achromatic (grey)
		rgb.x = rgb.y = rgb.z = v;
		return rgb;
	}
	h /= 60;			// sector 0 to 5
	i = floor( h );
	f = h - i;			// factorial part of h
	p = v * ( 1 - s );
	q = v * ( 1 - s * f );
	t = v * ( 1 - s * ( 1 - f ) );
	switch( i ) {
		case 0:
			rgb.x = v;
			rgb.y = t;
			rgb.z = p;
			break;
		case 1:
			rgb.x = q;
			rgb.y = v;
			rgb.z = p;
			break;
		case 2:
			rgb.x = p;
			rgb.y = v;
			rgb.z = t;
			break;
		case 3:
			rgb.x = p;
			rgb.y = q;
			rgb.z = v;
			break;
		case 4:
			rgb.x = t;
			rgb.y = p;
			rgb.z = v;
			break;
		default:		// case 5:
			rgb.x = v;
			rgb.y = p;
			rgb.z = q;
			break;
	}
	return rgb;
}


uchar glVal_to_cl(uint val) {
	if (val >= 14340) return round(0.1245790*val - 1658.44);	//>=128
	if (val >= 13316) return round(0.0622869*val - 765.408);
	if (val >= 12292) return round(0.0311424*val - 350.800);
	if (val >= 11268) return round(0.0155702*val - 159.443);

	float v = (float) val;
	return round(0.0000000000000125922*pow(v,4.f) - 0.00000000026729*pow(v,3.f) + 0.00000198135*pow(v,2.f) - 0.00496681*v - 0.0000808829); 
}