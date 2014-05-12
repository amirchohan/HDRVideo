#include "Filter.h"

namespace hdr
{
class ReinhardGlobal : public Filter {
public:
	ReinhardGlobal(float _key=0.18f, float _sat=1.6f);

	virtual bool setupOpenCL(cl_context_properties context_prop[], const Params& params);
	virtual double runCLKernels();
	virtual bool runOpenCL(int input_texid, int output_texid);
	virtual bool runOpenCL(Image input, Image output);
	virtual bool cleanupOpenCL();
	virtual bool runReference(Image input, Image output);

protected:
	//some parameters
	float key;
	float sat;
};
}