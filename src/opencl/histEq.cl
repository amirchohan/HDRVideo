
float3 RGBtoHSV(float r, float g, float b);
uchar3 HSVtoRGB(float h, float s, float v);

//computes the histogram for brightness
kernel void brightness_partial_hist( __global unsigned char* image,
							__global unsigned int* partial_histogram,
							__global unsigned int* brightness_hist) {

	const int gid = get_global_id(0);

	if (gid < image_size) {
		const int b = max(max(image[gid*3 + 0], image[gid*3 + 1]), image[gid*3 + 2]);
		const int tid = get_local_id(0);
		local uint local_hist[256];

		if(tid < 256) local_hist[tid] = 0;	//this will only work if local size >= 256

		barrier(CLK_LOCAL_MEM_FENCE);
		atomic_inc(&local_hist[b]);

		barrier(CLK_LOCAL_MEM_FENCE);	//this will only work if local size >= 256
		if (tid < 256) {
			partial_histogram[256*get_group_id(0) + tid] = local_hist[tid];
			brightness_hist[tid] = 0;
		}
	}
	return;
}

kernel void brightness_hist(__global unsigned int* partial_histogram,
							__global unsigned int* brightness_hist) {
	const int gid = get_global_id(0);
	unsigned int sum = partial_histogram[256 + gid];

	//num_workgroups is the number of workgroups in the previous kernel
	for (int i = 1; i < num_workgroups; i++) {
		sum += partial_histogram[256*i + gid];
	}
	brightness_hist[gid] = sum;
}

//computes the cdf of the brightness histogram
kernel void hist_cdf( __global unsigned int* hist) {
	const int gid = get_global_id(0);
	const int global_size = get_global_size(0);

	if (gid==0)
		for (int i=1; i<global_size; i++) {
			hist[i] += hist[i-1];
		}
	return;
}

//kernel to modify the brightness in the original image
kernel void modify_brightness( __global unsigned char* image,
							__global unsigned int* brightness_cdf) {
	const int i = get_global_id(0);

	if (i < image_size) {
		float r = image[i*3 + 0];	//Red
		float g = image[i*3 + 1];	//Green
		float b = image[i*3 + 2];	//Blue
		float3 hsv = RGBtoHSV(r, g, b);		//Convert to HSV to get Hue and Saturation

		hsv.z = (255*(brightness_cdf[(int)hsv.z] - brightness_cdf[0]))
					/(image_size - brightness_cdf[0]);

		uchar3 rgb = HSVtoRGB(hsv.x, hsv.y, hsv.z);	//Convert back to RGB with the modified brightness for V
		image[i*3 + 0] = rgb.x;
		image[i*3 + 1] = rgb.y;
		image[i*3 + 2] = rgb.z;	
	}
	return;
}

float3 RGBtoHSV(float r, float g, float b) {
	float rgb_min = min(min(r, g), b);
	float rgb_max = max(max(r, g), b);

	float hue, sat, val;

	//Value
	val = rgb_max;

	//Saturation
	float delta = rgb_max - rgb_min;
	if(rgb_max != 0) sat = delta/rgb_max;
	else {	// r = g = b = 0	//Saturation = 0, Value is undefined
		return (float3) (-1, 0, 0);
	}

	//Hue
	if(r == rgb_max) 		hue = (g-b)/delta;
	else if(g == rgb_max) 	hue = (b-r)/delta + 2;
	else 					hue = (r-g)/delta + 4;
	hue *= 60;				
	if(hue < 0) hue += 360;

	return (float3) (hue, sat, val);
}

uchar3 HSVtoRGB(float h, float s, float v) {
	uchar r, g, b;
	if( s == 0 ) { // achromatic (grey)
		return (uchar3) (v, v, v);
	}
	int i;
	float f, p, q, t;
	h /= 60;			// sector 0 to 5
	i = floor( h );
	f = h - i;			// factorial part of h
	p = v * ( 1 - s );
	q = v * ( 1 - s * f );
	t = v * ( 1 - s * ( 1 - f ) );
	switch( i ) {
		case 0:
			r = v;
			g = t;
			b = p;
			break;
		case 1:
			r = q;
			g = v;
			b = p;
			break;
		case 2:
			r = p;
			g = v;
			b = t;
			break;
		case 3:
			r = p;
			g = q;
			b = v;
			break;
		case 4:
			r = t;
			g = p;
			b = v;
			break;
		default:		// case 5:
			r = v;
			g = p;
			b = q;
			break;
	}
	return (uchar3) (r, g, b);
}