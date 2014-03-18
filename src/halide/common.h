#include <Halide.h>

using namespace Halide;
using namespace std;

Expr u8(Expr x)
{
  return cast(UInt(8), x);
}

Expr u16(Expr x)
{
  return cast(UInt(16), x);
}

Expr u32(Expr x)
{
  return cast(UInt(16), x);
}

Expr s8(Expr x)
{
  return cast(Int(8), x);
}

Expr s16(Expr x)
{
  return cast(Int(16), x);
}

Expr s32(Expr x)
{
  return cast(Int(16), x);
}

Expr f16(Expr x)
{
  return cast(Float(16), x);
}

Expr f32(Expr x)
{
  return cast(Float(32), x);
}

void compile(Func func, ImageParam input, string fnName, string prefix)
{
  vector<Argument> args;
  args.push_back(input);
  func.compile_to_assembly(prefix+".s", args, fnName);
  func.compile_to_header(prefix+".h", args, fnName);
}
