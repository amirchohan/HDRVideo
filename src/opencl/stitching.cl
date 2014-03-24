
float weight(float luminance);
float getPixelLuminance(float3 pixel_val);

kernel void stitch( __global float* LDRimages, __global float* exposures, __global float* HDRimage) {
	//this kernel takes num_images LDR images in input_images
	//and combines them together according to their exposures

	int gid = get_global_id(0);	//id in the entire global memory

	if (gid < image_size) {
	float weightedSum = 0;
	float3 hdr, ldr;
	hdr.x = hdr.y = hdr.z = 0;
	for (int i=0; i < num_images; i++) {
		ldr.x = LDRimages[i*image_size*4 + (gid*4 + 0)]*255.f;
		ldr.y = LDRimages[i*image_size*4 + (gid*4 + 1)]*255.f;
		ldr.z = LDRimages[i*image_size*4 + (gid*4 + 2)]*255.f;

		float luminance = getPixelLuminance(ldr);
		float w = weight(luminance);
		float exposure = exposures[i];

		hdr.x += (ldr.x/exposure) * w;
		hdr.y += (ldr.y/exposure) * w;
		hdr.z += (ldr.z/exposure) * w;

		weightedSum += w;
	}

	hdr.x = hdr.x/(weightedSum + 0.000001);
	hdr.y = hdr.y/(weightedSum + 0.000001);
	hdr.z = hdr.z/(weightedSum + 0.000001);

	HDRimage[gid*4 + 0] = hdr.x/255.f;
	HDRimage[gid*4 + 1] = hdr.y/255.f;
	HDRimage[gid*4 + 2] = hdr.z/255.f;
	HDRimage[gid*4 + 3] = getPixelLuminance(hdr);
	}
}

float weight(float luminance) {
	if (luminance < 0.5) return luminance*2.0;
	else return (1.0 - luminance)*2.0;
}

float getPixelLuminance(float3 pixel_val) {
	return pixel_val.x*0.2126 + pixel_val.y*0.7152 + pixel_val.z*0.0722;
}