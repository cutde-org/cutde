import logging
import os
import warnings

import pyopencl
import pyopencl.array

from .gpu_backend import load

logger = logging.getLogger(__name__)

gpu_initialized = False
gpu_ctx = None
gpu_queue = None


def report_devices(ctx):
    device_names = [d.name for d in ctx.devices]
    logger.info("initializing opencl context with devices = " + str(device_names))


def initialize_with_ctx(ctx):
    global gpu_initialized, gpu_ctx, gpu_queue
    gpu_ctx = ctx
    gpu_queue = pyopencl.CommandQueue(gpu_ctx)
    gpu_initialized = True

    report_devices(ctx)


def avoid_apple_cpu(ctx):
    """
    The Apple CPU OpenCL implementation is awful. Instead, we should just use
    PoCL.
    """
    if ctx.devices[0].platform.name == "Apple" and "CPU" in ctx.devices[0].name:
        platforms = pyopencl.get_platforms()
        platform_idx = None
        for i, p in enumerate(platforms):
            if p.name != "Apple":
                platform_idx = i
            else:
                apple_platform_idx = i
        if platform_idx is not None:
            warnings.warn(
                "The OpenCL context created used the Apple CPU"
                " implementation which is not supported. Trying again"
                f" with a different platform: {p.name}"
            )
            return pyopencl.create_some_context(answers=[str(platform_idx)])

        # If no other platforms were found, let's try to
        # find a non-CPU device like an Iris Pro.
        platform_idx = apple_platform_idx
        device_idx = None
        for i, d in enumerate(platforms[platform_idx].get_devices()):
            if "CPU" in d.name:
                continue
            device_idx = i
            break

        if device_idx is not None:
            warnings.warn(
                "The OpenCL context created used the Apple CPU"
                " implementation which is not supported. Trying again"
                f" with a different device: {d.name}"
            )
            return pyopencl.create_some_context(
                answers=[str(platform_idx), str(device_idx)]
            )

        raise NotImplementedError(
            "cutde does not support the Apple CPU OpenCL implementation and no other"
            " platform or device was found. Please consult the cutde README"
        )
    return ctx


def ensure_initialized():
    if not gpu_initialized:
        ctx = pyopencl.create_some_context()
        ctx = avoid_apple_cpu(ctx)
        initialize_with_ctx(ctx)  # this updates gpu_initialized


def ptr(arr):
    if type(arr) is pyopencl.array.Array:
        return arr.data
    return arr


def to(arr, float_type):
    ensure_initialized()
    if type(arr) is pyopencl.array.Array:
        return arr
    to_type = arr.astype(float_type)
    return pyopencl.array.to_device(gpu_queue, to_type)


def zeros(shape, float_type):
    ensure_initialized()
    return pyopencl.array.zeros(gpu_queue, shape, float_type)


def empty(shape, float_type):
    ensure_initialized()
    return pyopencl.array.empty(gpu_queue, shape, float_type)


def get(arr):
    return arr.get()


class ModuleWrapper:
    def __init__(self, module):
        self.module = module

    def __getattr__(self, name):
        kernel = getattr(self.module, name)

        def provide_queue_wrapper(*args):
            grid = args[-2]
            block = args[-1]
            global_size = [b * g for b, g in zip(grid, block)]
            arg_ptrs = [ptr(a) for a in args[:-2]]
            return kernel(gpu_queue, global_size, block, *arg_ptrs)

        return provide_queue_wrapper


def compile(code):
    ensure_initialized()

    compile_options = []

    if "CUTDE_OPENCL_NO_OPTS" in os.environ:
        logger.debug("compiling opencl without optimizations.")
        compile_options.append("-cl-opt-disable")

    return ModuleWrapper(pyopencl.Program(gpu_ctx, code).build(options=compile_options))


def max_block_size(requested):
    return requested


def load_module(
    tmpl_name, tmpl_dir=None, save_code=False, no_caching=False, tmpl_args=None
):
    return load(
        "opencl",
        preamble,
        compile,
        tmpl_name,
        tmpl_dir,
        save_code,
        no_caching,
        tmpl_args,
    )


preamble = """
// taken from pyopencl._cluda
// 'static' helps to avoid the "no previous prototype for function" warning
#if __OPENCL_VERSION__ >= 120
#define WITHIN_KERNEL static
#else
#define WITHIN_KERNEL
#endif
#define KERNEL __kernel
#define GLOBAL_MEM __global
#define LOCAL_MEM __local
#define LOCAL_MEM_DYNAMIC __local
#define LOCAL_MEM_ARG __local
#define CONSTANT __constant
// INLINE is already defined in Beignet driver
#ifndef INLINE
#define INLINE inline
#endif
#define SIZE_T size_t
#define VSIZE_T size_t
// used to align fields in structures
#define ALIGN(bytes) __attribute__ ((aligned(bytes)))
#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64: enable
#endif
"""
