#include "Filter.h"

namespace hdr
{
class GammaAdj : public Filter {
public:
	GammaAdj();

	virtual bool runHalideCPU(LDRI input, Image output, const Params& params);
	virtual bool runHalideGPU(LDRI input, Image output, const Params& params);
	virtual bool runOpenCL(LDRI input, Image output, const Params& params);
	virtual bool runReference(LDRI input, Image output);
};
}
