#include "common.h"
#include <iostream>

int main(int argc, char *argv[])
{
  if (argc != 4)
  {
    cout << "Usage: " << argv[0] << " cpu|gpu out_func out_prefix" << endl;
    return 1;
  }

  ImageParam input(UInt(8), 3, "input");
  Func clamped("clamped");
  Func coeff("coeff"), sum("sum");
  Func weight("weight");
  Func bilateral("bilateral");
  Var c("c"), x("x"), y("y"), i("i"), j("j");

  // Algorithm
  clamped(x, y, c) = input(
    clamp(x, 0, input.width()-1),
    clamp(y, 0, input.height()-1),
    c) / 255.f;

  Expr imgDist = (sqrt(f32(i*i) + f32(j*j))) * (1.f/3.f);
  Expr colDist = sqrt(
    pow(clamped(x+i, y+j, 0)-clamped(x, y, 0), 2) +
    pow(clamped(x+i, y+j, 1)-clamped(x, y, 1), 2) +
    pow(clamped(x+i, y+j, 2)-clamped(x, y, 2), 2)
  ) * (1.f/0.2f);

  weight(x, y, i, j) =
    exp(-0.5f * (imgDist*imgDist)) *
    exp(-0.5f * (colDist*colDist));

  RDom r(-2, 5, -2, 5, "r");
  coeff(x, y) += weight(x, y, r.x, r.y);
  sum(x, y, c) += weight(x, y, r.x, r.y) * clamped(x+r.x, y+r.y, c);

  bilateral(x, y, c) =
    select(
      c==3,
      255,
      u8(clamp(sum(x, y, c)/coeff(x, y), 0, 1) * 255)
    );

  // Channel order
  input.set_stride(0, 4);
  input.set_extent(2, 4);
  bilateral.reorder_storage(c, x, y);
  bilateral.output_buffer().set_stride(0, 4);
  bilateral.output_buffer().set_extent(2, 4);

  // Schedules
  if (!strcmp(argv[1], "cpu"))
  {
    bilateral.parallel(y);
  }
  else if (!strcmp(argv[1], "gpu"))
  {
    bilateral.cuda_tile(x, y, 16, 4);
  }
  else
  {
    cout << "Invalid schedule type '" << argv[1] << "'" << endl;
    return 1;
  }

  compile(bilateral, input, argv[2], argv[3]);

  return 0;
}
