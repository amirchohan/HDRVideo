#include "Filter.h"

namespace hdr
{
class Stitching : public Filter {
public:
	Stitching();

	virtual bool runHalideCPU(LDRI input, Image output, const Params& params);
	virtual bool runHalideGPU(LDRI input, Image output, const Params& params);
	virtual bool runOpenCL(LDRI input, Image output, const Params& params);
	virtual bool runReference(LDRI input, Image output);
};

float weight(float luminance);

}
