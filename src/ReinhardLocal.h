#include "Filter.h"

namespace hdr
{
class ReinhardLocal : public Filter {
public:
	ReinhardLocal(float _key=0.18f, float _sat=1.6f, float _epsilon=0.05, float _phi=8.0);

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
	float epsilon;
	float phi;

};
}