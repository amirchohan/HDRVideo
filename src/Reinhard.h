#include "Filter.h"

namespace hdr
{
class Reinhard : public Filter {
public:
	Reinhard();

	virtual bool runHalideCPU(Image input, Image output, const Params& params);
	virtual bool runHalideGPU(Image input, Image output, const Params& params);
	virtual bool runOpenCL(Image input, Image output, const Params& params);
	virtual bool runReference(Image input, Image output);
};

float weight(float luminance);

}
