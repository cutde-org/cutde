import os
import warnings
from math import ceil

import numpy as np

import cutde.backend as backend

source_dir = os.path.dirname(os.path.realpath(__file__))

DISP_FS = ("disp_fs", 3)
STRAIN_FS = ("strain_fs", 6)
DISP_HS = ("disp_hs", 3)
STRAIN_HS = ("strain_hs", 6)


class Placeholder:
    pass


placeholder = Placeholder()


def solve_types(obs_pts, tris, slips):
    type_map = {
        np.int32: np.float32,
        np.int64: np.float64,
        np.float32: np.float32,
        np.float64: np.float64,
    }

    float_type = None
    out_arrs = []
    for name, arr in [("obs_pts", obs_pts), ("tris", tris), ("slips", slips)]:
        if isinstance(arr, Placeholder):
            out_arrs.append(arr)
            continue

        dtype = arr.dtype.type

        if dtype not in type_map:
            raise ValueError(
                f"The {name} input array has type {arr.dtype} but must have a float or"
                " integer dtype."
            )

        if float_type is None:
            float_type = type_map[dtype]

            # If we're using OpenCL, we need to check if float64 is allowed.
            # If not, convert to float32.
            if backend.which_backend == "opencl":
                import cutde.opencl

                cutde.opencl.ensure_initialized()
                extensions = (
                    cutde.opencl.gpu_ctx.devices[0].extensions.strip().split(" ")
                )
                if "cl_khr_fp64" not in extensions and float_type is np.float64:
                    warnings.warn(
                        "The OpenCL implementation being used does not support "
                        "float64. This will require converting arrays to float32."
                    )
                    float_type = np.float32

        out_arr = arr
        if dtype != float_type:
            warnings.warn(
                f"The {name} input array has type {out_arr.dtype} but needs "
                "to be converted"
                f" to dtype {np.dtype(float_type)}. Converting {name} to "
                f"{np.dtype(float_type)} may be expensive."
            )
            out_arr = out_arr.astype(float_type)

        if not out_arr.flags.c_contiguous:
            warnings.warn(
                f"The {name} input array is not C-contiguous. "
                "Converting to C ordering. This may be expensive."
            )
            out_arr = np.ascontiguousarray(out_arr)

        out_arrs.append(out_arr)

    return float_type, out_arrs


def check_inputs(obs_pts, tris, slips):
    if obs_pts.shape[1] != 3:
        raise ValueError(
            "The second dimension of the obs_pts array must be 3 because the "
            "observation points should be locations in three-dimensional space."
        )
    if tris.shape[1] != 3:
        raise ValueError(
            "The second dimension of the tris array must be 3 because there must be "
            "three vertices per triangle."
        )
    if tris.shape[2] != 3:
        raise ValueError(
            "The third dimension of the tris array must be 3 because the triangle "
            "vertices should be locations in three-dimensional space."
        )
    if not isinstance(slips, Placeholder) and (slips.shape[0] != tris.shape[0]):
        raise ValueError(
            "The number of input slip vectors must be equal to the number of input"
            " triangles."
        )
    if not isinstance(slips, Placeholder) and (slips.shape[1] != 3):
        raise ValueError(
            "The second dimension of the slips array must be 3 because each row "
            "should be a vector in the TDE coordinate system (strike-slip, dip-slip,"
            " tensile-slip)."
        )


def call_clu(obs_pts, tris, slips, nu, fnc):
    fnc_name, vec_dim = fnc
    if tris.shape[0] != obs_pts.shape[0]:
        raise ValueError("There must be one input observation point per triangle.")

    check_inputs(obs_pts, tris, slips)
    float_type, (obs_pts, tris, slips) = solve_types(obs_pts, tris, slips)

    n = obs_pts.shape[0]
    block_size = backend.max_block_size(16)
    n_blocks = int(np.ceil(n / block_size))
    gpu_config = dict(
        block_size=block_size, float_type=backend.np_to_c_type(float_type)
    )
    module = backend.load_module("pairs.cu", tmpl_args=gpu_config, tmpl_dir=source_dir)

    gpu_results = backend.empty(n * vec_dim, float_type)
    gpu_obs_pts = backend.to(obs_pts, float_type)
    gpu_tris = backend.to(tris, float_type)
    gpu_slips = backend.to(slips, float_type)

    getattr(module, "pairs_" + fnc_name)(
        gpu_results,
        np.int32(n),
        gpu_obs_pts,
        gpu_tris,
        gpu_slips,
        float_type(nu),
        (n_blocks, 1, 1),
        (block_size, 1, 1),
    )
    out = backend.get(gpu_results).reshape((n, vec_dim))
    return out


def call_clu_matrix(obs_pts, tris, nu, fnc):
    fnc_name, vec_dim = fnc
    check_inputs(obs_pts, tris, placeholder)
    float_type, (obs_pts, tris, _) = solve_types(obs_pts, tris, placeholder)

    n_obs = obs_pts.shape[0]
    n_src = tris.shape[0]
    block_size = backend.max_block_size(16)
    n_obs_blocks = int(np.ceil(n_obs / block_size))
    n_src_blocks = int(np.ceil(n_src / block_size))
    gpu_config = dict(float_type=backend.np_to_c_type(float_type))
    module = backend.load_module("matrix.cu", tmpl_args=gpu_config, tmpl_dir=source_dir)

    gpu_results = backend.empty(n_obs * vec_dim * n_src * 3, float_type)
    gpu_obs_pts = backend.to(obs_pts, float_type)
    gpu_tris = backend.to(tris, float_type)

    getattr(module, "matrix_" + fnc_name)(
        gpu_results,
        np.int32(n_obs),
        np.int32(n_src),
        gpu_obs_pts,
        gpu_tris,
        float_type(nu),
        (n_obs_blocks, n_src_blocks, 1),
        (block_size, block_size, 1),
    )
    out = backend.get(gpu_results).reshape((n_obs, vec_dim, n_src, 3))
    return out


def call_clu_free(obs_pts, tris, slips, nu, fnc):
    fnc_name, vec_dim = fnc
    check_inputs(obs_pts, tris, slips)
    float_type, (obs_pts, tris, slips) = solve_types(obs_pts, tris, slips)

    n_obs = obs_pts.shape[0]
    n_src = tris.shape[0]
    block_size = backend.max_block_size(256)

    gpu_obs_pts = backend.to(obs_pts, float_type)
    gpu_tris = backend.to(tris, float_type)
    gpu_slips = backend.to(slips, float_type)
    gpu_results = backend.zeros(n_obs * vec_dim, float_type)

    n_obs_blocks = int(np.ceil(n_obs / block_size))
    gpu_config = dict(float_type=backend.np_to_c_type(float_type))
    module = backend.load_module("free.cu", tmpl_args=gpu_config, tmpl_dir=source_dir)

    # Split up the sources into chunks so that we don't completely overwhelm a
    # single GPU machine and cause the screen to lock up.
    default_chunk_size = 64
    n_chunks = int(ceil(n_src / default_chunk_size))
    out = np.zeros((n_obs, vec_dim), dtype=float_type)
    for i in range(n_chunks):
        chunk_start = i * default_chunk_size
        chunk_size = min(n_src - chunk_start, default_chunk_size)
        chunk_end = chunk_start + chunk_size

        getattr(module, "free_" + fnc_name)(
            gpu_results,
            np.int32(n_obs),
            np.int32(n_src),
            np.int32(chunk_start),
            np.int32(chunk_end),
            gpu_obs_pts,
            gpu_tris,
            gpu_slips,
            float_type(nu),
            (n_obs_blocks, 1, 1),
            (block_size, 1, 1),
        )
        out += backend.get(gpu_results).reshape((n_obs, vec_dim))
    return out


def process_block_inputs(obs_start, obs_end, src_start, src_end):
    out_arrs = []
    for name, a in [
        ("obs_start", obs_start),
        ("obs_end", obs_end),
        ("src_start", src_start),
        ("src_end", src_end),
    ]:
        a = np.array(a)
        if a.shape[0] != np.array(obs_start).shape[0]:
            raise ValueError(f"The length of {name} must match obs_start.")
        if not (a.dtype.type is np.int32 or a.dtype.type is np.int64):
            raise ValueError(f"The {name} array must have integer type.")

        # Don't bother warning for these conversions since the cost of
        # converting a single value per block is tiny.
        if a.flags.f_contiguous or a.dtype.type != np.int32:
            a = np.ascontiguousarray(a, dtype=np.int32)

        out_arrs.append(a)

    return out_arrs


def call_clu_block(obs_pts, tris, obs_start, obs_end, src_start, src_end, nu, fnc):
    fnc_name, vec_dim = fnc
    check_inputs(obs_pts, tris, placeholder)
    float_type, (obs_pts, tris, _) = solve_types(obs_pts, tris, placeholder)
    obs_start, obs_end, src_start, src_end = process_block_inputs(
        obs_start, obs_end, src_start, src_end
    )

    block_sizes = vec_dim * 3 * (obs_end - obs_start) * (src_end - src_start)
    block_end = np.cumsum(block_sizes)
    block_start = np.empty(block_end.shape[0] + 1, dtype=block_end.dtype)
    block_start[:-1] = block_end - block_sizes
    block_start[-1] = block_end[-1]

    n_blocks = obs_end.shape[0]
    team_size = backend.max_block_size(16)
    gpu_config = dict(float_type=backend.np_to_c_type(float_type))
    module = backend.load_module("blocks.cu", tmpl_args=gpu_config, tmpl_dir=source_dir)

    gpu_results = backend.zeros(block_end[-1], float_type)
    gpu_obs_pts = backend.to(obs_pts, float_type)
    gpu_tris = backend.to(tris, float_type)
    gpu_obs_start = backend.to(obs_start, np.int32)
    gpu_obs_end = backend.to(obs_end, np.int32)
    gpu_src_start = backend.to(src_start, np.int32)
    gpu_src_end = backend.to(src_end, np.int32)
    gpu_block_start = backend.to(block_start, np.int32)

    getattr(module, "blocks_" + fnc_name)(
        gpu_results,
        gpu_obs_pts,
        gpu_tris,
        gpu_obs_start,
        gpu_obs_end,
        gpu_src_start,
        gpu_src_end,
        gpu_block_start,
        float_type(nu),
        (n_blocks, 1, 1),
        (team_size, 1, 1),
    )
    return backend.get(gpu_results), block_start
