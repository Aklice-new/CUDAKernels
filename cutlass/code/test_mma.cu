#include "cute/atom/copy_atom.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/pointer_flagged.hpp"
#include "cute/swizzle_layout.hpp"
#include "cute/tensor_impl.hpp"
#include "cute/util/print.hpp"
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"

int main()
{
    using namespace cute;
    auto inner_shape1 = make_shape(2, 2);
    auto inner_shape2 = make_shape(2, 4);
    auto shape = make_shape(inner_shape1, inner_shape2);
    auto inner_strides1 = make_shape(1, 4);
    auto inner_strides2 = make_shape(2, 8);
    auto strides = make_shape(inner_strides1, inner_strides2);
    auto layout = make_layout(shape, strides);
    // print_latex(layout);
    print(size<0>(layout));
    print('\n');
    print(size<1>(layout));
    using namespace cute;
    // Nonowning Tensor
    auto layout2 = make_layout(make_shape(4, 8), make_shape(1, 4));
    auto data = std::vector<int>(32);
    auto tensor = make_tensor(data.data(), layout);
    print(" Nonowning Tensor : ");
    print(tensor);
    // print_latex(tensor.layout());
    auto tile = local_tile(tensor, make_shape(2, 2), make_shape(1, 4));
    print(" \nTile : ");
    print(tile);
    // print_latex(tile.layout());
}