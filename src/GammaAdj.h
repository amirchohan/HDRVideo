#include "Filter.h"

namespace hdr
{
class GammaAdj : public Filter {
public:
	GammaAdj();

	virtual bool runHalideCPU(Image input, Image output, const Params& params);
	virtual bool runHalideGPU(Image input, Image output, const Params& params);
	virtual bool runOpenCL(Image input, Image output, const Params& params);
	virtual bool runReference(Image input, Image output);
};
}
