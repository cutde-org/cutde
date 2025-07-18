<%!
def comp(d):
    return ["x", "y", "z", "a", "b", "c"][d]
%>
<%namespace name="common" file="common.cu"/>

#include <math.h>
#include <cstdio>
#include <algorithm>

#define WITHIN_KERNEL 
#define KERNEL
#define GLOBAL_MEM
#define LOCAL_MEM

using std::min;

struct XYZ {
    int x;
    int y;
    int z;
};

thread_local XYZ blockIdx;
XYZ gridDim;

WITHIN_KERNEL int get_local_id(unsigned int dim) { return 0; }

WITHIN_KERNEL int get_group_id(unsigned int dim)
{
    if(dim == 0) return blockIdx.x;
    if(dim == 1) return blockIdx.y;
    if(dim == 2) return blockIdx.z;
    return 0;
}

WITHIN_KERNEL int get_local_size(unsigned int dim) { return 1; }

WITHIN_KERNEL int get_num_groups(unsigned int dim)
{
    if(dim == 0) return gridDim.x;
    if(dim == 1) return gridDim.y;
    if(dim == 2) return gridDim.z;
    return 1;
}
WITHIN_KERNEL int get_global_size(unsigned int dim)
{
    return get_num_groups(dim);
}
WITHIN_KERNEL int get_global_id(unsigned int dim)
{
    return get_group_id(dim);
}

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;

<%include file="pairs.cu"/>
<%include file="blocks.cu"/>
<%include file="matrix.cu"/>
<%include file="free.cu"/>
<%include file="aca.cu"/>

template <typename T>
T conv_arg(T arg) {
    return arg;
}

template <typename T>
T* conv_arg(py::array_t<T> arg) {
    return arg.mutable_data(0);
}

template <typename T>
struct pyarg_from_cpparg {
    using PyArgType = T;
};

template <typename T>
struct pyarg_from_cpparg<T*> {
    using PyArgType = py::array_t<T>;
};

template <typename R, typename ...Args>
decltype(auto) wrapper(R(*fn)(Args...))
{
    return [=](typename pyarg_from_cpparg<Args>::PyArgType... args, 
             std::tuple<int,int,int> grid,
             std::tuple<int,int,int> block) 
    {
        gridDim = {std::get<0>(grid), std::get<1>(grid), std::get<2>(grid)};
        blockIdx = {0,0,0};

        int Ngrid = gridDim.x * gridDim.y * gridDim.z;

        auto ptr_args = std::make_tuple(conv_arg(args)...);

        #pragma omp parallel for
        for (long i = 0; i < Ngrid; i++) {
            long i_r = i;
            blockIdx.z = i_r % gridDim.z;
            i_r /= gridDim.z;
            blockIdx.y = i_r % gridDim.y;
            i_r /= gridDim.y;
            blockIdx.x = i_r % gridDim.x;
            i_r /= gridDim.x;

            std::apply(fn, ptr_args);
        }
    };
}

PYBIND11_MODULE(cpp_backend_${float_type}, m) {
    % for type in ["pairs", "blocks", "matrix", "free", "aca"]:
        % for field in ["disp", "strain"]:
            % for space in ["fs", "hs"]:
                <%
                fnc_name = type + '_' + field + '_' + space
                %>
                m.def("${fnc_name}", wrapper(${fnc_name}));
            % endfor
        % endfor
    % endfor
}
