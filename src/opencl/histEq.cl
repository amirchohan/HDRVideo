

float3 RGBtoHSV(float3 rgb);
float3 HSVtoRGB(float3 hsv);

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
		brightness = max(max(image[i*NUM_CHANNELS + 0], image[i*NUM_CHANNELS + 1]), image[i*NUM_CHANNELS + 2])*HIST_SIZE;
		barrier(CLK_LOCAL_MEM_FENCE);
		atomic_inc(&l_hist[brightness]);
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	for (int i = lid; i < HIST_SIZE; i+=group_size) {
		partial_histogram[i + group_id * HIST_SIZE] = l_hist[i];
	}
}

kernel void merge_hist(__global uint* partial_histogram, __global uint* histogram, __local uint* l_Data, const int num_hists) {

	const int group_size = get_local_size(0);
	const int lid = get_local_id(0);
	const int group_id = get_group_id(0);

	uint sum = 0;
	for(uint i = lid; i < num_hists; i += group_size)
		sum += partial_histogram[group_id + i * HIST_SIZE];

	l_Data[lid] = sum;
	for(uint stride = group_size/2; stride > 0; stride >>= 1){
		barrier(CLK_LOCAL_MEM_FENCE);
		if(lid < stride) l_Data[lid] += l_Data[lid + stride];
	}

	if(lid == 0) histogram[group_id] = l_Data[0];
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
kernel void histogram_equalisation( __global float* image, __global uint* brightness_cdf) {
	for (int i= get_global_id(0); i < image_size; i+=get_global_size(0)) {
		float3 rgb = (float3) (image[i*NUM_CHANNELS + 0], image[i*NUM_CHANNELS + 1], image[i*NUM_CHANNELS + 2]);
		float3 hsv = RGBtoHSV(rgb);		//Convert to HSV to get Hue and Saturation

		hsv.z = ((HIST_SIZE-1)*(brightness_cdf[(int)hsv.z] - brightness_cdf[0]))
					/(image_size - brightness_cdf[0]);

		rgb = HSVtoRGB(hsv);	//Convert back to RGB with the modified brightness for V
		image[i*NUM_CHANNELS + 0] = rgb.x;
		image[i*NUM_CHANNELS + 1] = rgb.y;
		image[i*NUM_CHANNELS + 2] = rgb.z;
	}
}

float3 RGBtoHSV(float3 rgb) {
	float r = rgb.x*HIST_SIZE;
	float g = rgb.y*HIST_SIZE;
	float b = rgb.z*HIST_SIZE;
	float rgb_min, rgb_max, delta;
	rgb_min = min(min(r, g), b);
	rgb_max = clamp(max(max(r, g), b), 0.f, HIST_SIZE*1.f-1);

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
	else 				hsv.x = (r-g)/delta + 4;
	hsv.x *= 60;				
	if( hsv.x < 0 ) hsv.x += 360;

	return hsv;
}

float3 HSVtoRGB(float3 hsv) {
	int i;
	float h = hsv.x;
	float s = hsv.y;
	float v = hsv.z;
	float f, p, q, t;
	float3 rgb;
	if( s == 0 ) { // achromatic (grey)
		rgb.x = rgb.y = rgb.z = v/HIST_SIZE;
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
	rgb.x = rgb.x/HIST_SIZE;
	rgb.y = rgb.y/HIST_SIZE;
	rgb.z = rgb.z/HIST_SIZE;
	return rgb;
}