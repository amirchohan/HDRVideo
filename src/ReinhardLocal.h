#include "Filter.h"

namespace hdr
{
class ReinhardLocal : public Filter {
public:
	ReinhardLocal(float _key=0.18f, float _sat=1.6f, float _epsilon=0.05, float _phi=8.0);

	virtual bool setupOpenCL(cl_context_properties context_prop[], const Params& params);
	virtual double runCLKernels(bool recomputeMapping);
	virtual bool runOpenCL(int input_texid, int output_texid, bool recomputeMapping);
	virtual bool runOpenCL(Image input, Image output, bool recomputeMapping);
	virtual bool cleanupOpenCL();
	virtual bool runReference(Image input, Image output);

protected:
	//some parameters
	float key;
	float sat;
	float epsilon;
	float phi;

	//information regarding all mipmap levels
	int num_mipmaps;
	int* m_width;		//at index i this contains the width of the mipmap at index i
	int* m_height;		//at index i this contains the height of the mipmap at index i
	int* m_offset;		//at index i this contains the start point to store the mipmap at level i

};
}