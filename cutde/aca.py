from math import ceil

import numpy as np

import cutde.backend as backend
from cutde.coordinators import (
    check_inputs,
    placeholder,
    process_block_inputs,
    solve_types,
    source_dir,
)


def check_tol_max_iter(obs_start, tol, max_iter, float_type):
    tol = np.array(tol)
    if tol.shape[0] != obs_start.shape[0]:
        raise ValueError("The length of tol must match obs_start.")

    tol = np.ascontiguousarray(tol, dtype=float_type)

    max_iter = np.array(max_iter)
    if max_iter.shape[0] != obs_start.shape[0]:
        raise ValueError("The length of max_iter must match obs_start.")

    max_iter = np.ascontiguousarray(max_iter, dtype=np.int32)
    return tol, max_iter


def call_clu_aca(
    obs_pts,
    tris,
    obs_start,
    obs_end,
    src_start,
    src_end,
    nu,
    tol,
    max_iter,
    fnc,
    Iref0=None,
    Jref0=None,
):
    fnc_name, vec_dim = fnc
    check_inputs(obs_pts, tris, placeholder)
    float_type, (obs_pts, tris, _) = solve_types(obs_pts, tris, placeholder)
    obs_start, obs_end, src_start, src_end = process_block_inputs(
        obs_start, obs_end, src_start, src_end
    )
    tol, max_iter = check_tol_max_iter(obs_start, tol, max_iter, float_type)

    default_chunk_size = 512
    team_size = backend.max_block_size(32)
    n_blocks = obs_end.shape[0]

    verbose = False
    gpu_config = dict(float_type=backend.np_to_c_type(float_type), verbose=verbose)
    module = backend.load_module("aca.cu", tmpl_args=gpu_config, tmpl_dir=source_dir)

    n_chunks = int(ceil(n_blocks / default_chunk_size))
    appxs = []
    for i in range(n_chunks):
        chunk_start = i * default_chunk_size
        chunk_size = min(n_blocks - chunk_start, default_chunk_size)
        chunk_end = chunk_start + chunk_size

        n_obs_per_block = (
            obs_end[chunk_start:chunk_end] - obs_start[chunk_start:chunk_end]
        )
        n_src_per_block = (
            src_end[chunk_start:chunk_end] - src_start[chunk_start:chunk_end]
        )
        n_rows = n_obs_per_block * vec_dim
        n_cols = n_src_per_block * 3
        block_sizes = n_rows * n_cols

        # Storage for the U, V output matrices. These will be in a packed format.
        gpu_buffer = backend.empty(block_sizes.sum(), float_type)

        # Storage for temporary rows and columns: RIref, RJref, RIstar, RJstar
        fworkspace_per_block = n_cols + n_rows + vec_dim * n_cols + 3 * n_rows
        fworkspace_ends = np.cumsum(fworkspace_per_block)
        fworkspace_starts = fworkspace_ends - fworkspace_per_block
        gpu_fworkspace = backend.empty(fworkspace_ends[-1], float_type)
        gpu_fworkspace_starts = backend.to(fworkspace_starts, np.int32)

        # uv_ptrs forms arrays that point to the start of each U/V vector pairs in
        # the main output buffer
        uv_ptrs_size = np.minimum(n_rows, n_cols)
        uv_ptrs_ends = np.cumsum(uv_ptrs_size)
        uv_ptrs_starts = uv_ptrs_ends - uv_ptrs_size
        gpu_uv_ptrs_starts = backend.to(uv_ptrs_starts, np.int32)
        gpu_uv_ptrs = backend.empty(uv_ptrs_ends[-1], np.int32)
        gpu_iworkspace = backend.empty(uv_ptrs_ends[-1], np.int32)

        # Output space for specifying the number of terms used for each
        # approximation.
        gpu_n_terms = backend.empty(chunk_size, np.int32)

        # Storage space for a pointer to the next empty portion of the output
        # buffer.
        gpu_next_ptr = backend.zeros(1, np.int32)

        # The index of the starting reference rows/cols.
        if Iref0 is None:
            Iref0_chunk = np.random.randint(0, n_rows, size=chunk_size, dtype=np.int32)
        else:
            Iref0_chunk = Iref0[chunk_start:chunk_end]
        if Jref0 is None:
            Jref0_chunk = np.random.randint(0, n_cols, size=chunk_size, dtype=np.int32)
        else:
            Jref0_chunk = Jref0[chunk_start:chunk_end]
        gpu_Iref0 = backend.to(Iref0_chunk, np.int32)
        gpu_Jref0 = backend.to(Jref0_chunk, np.int32)

        gpu_obs_pts = backend.to(obs_pts, float_type)
        gpu_tris = backend.to(tris, float_type)
        gpu_obs_start = backend.to(obs_start[chunk_start:chunk_end], np.int32)
        gpu_obs_end = backend.to(obs_end[chunk_start:chunk_end], np.int32)
        gpu_src_start = backend.to(src_start[chunk_start:chunk_end], np.int32)
        gpu_src_end = backend.to(src_end[chunk_start:chunk_end], np.int32)
        gpu_tol = backend.to(tol[chunk_start:chunk_end], float_type)
        gpu_max_iter = backend.to(max_iter[chunk_start:chunk_end], np.int32)

        if verbose:
            print(f"gpu_buffer.shape = {gpu_buffer.shape}")
            print(f"gpu_uv_ptrs.shape = {gpu_uv_ptrs.shape}")
            print(f"gpu_n_terms.shape = {gpu_n_terms.shape}")
            print(f"gpu_next_ptr.shape = {gpu_next_ptr.shape}")
            print(f"gpu_fworkspace.shape = {gpu_fworkspace.shape}")
            print(f"gpu_iworkspace.shape = {gpu_iworkspace.shape}")
            print(f"gpu_uv_ptrs_starts.shape = {gpu_uv_ptrs_starts.shape}")
            print(f"gpu_Iref0.shape = {gpu_Iref0.shape}")
            print(f"gpu_Jref0.shape = {gpu_Jref0.shape}")
            print(f"obs_pts.shape = {obs_pts.shape}")
            print(f"tris.shape = {tris.shape}")
            print(f"gpu_obs_start.shape = {gpu_obs_start.shape}")
            print(f"gpu_obs_end.shape = {gpu_obs_end.shape}")
            print(f"gpu_src_start.shape = {gpu_src_start.shape}")
            print(f"gpu_src_end.shape = {gpu_src_end.shape}")

        getattr(module, "aca_" + fnc_name)(
            gpu_buffer,
            gpu_uv_ptrs,
            gpu_n_terms,
            gpu_next_ptr,
            gpu_fworkspace,
            gpu_iworkspace,
            gpu_uv_ptrs_starts,
            gpu_fworkspace_starts,
            gpu_Iref0,
            gpu_Jref0,
            gpu_obs_pts,
            gpu_tris,
            gpu_obs_start,
            gpu_obs_end,
            gpu_src_start,
            gpu_src_end,
            gpu_tol,
            gpu_max_iter,
            float_type(nu),
            (chunk_size, 1, 1),
            (team_size, 1, 1),
        )

        # post-process the buffer to collect the U, V vectors
        buffer = backend.get(gpu_buffer)
        uv_ptrs = backend.get(gpu_uv_ptrs)
        n_terms = backend.get(gpu_n_terms)
        for i in range(chunk_size):
            us = []
            vs = []
            uv_ptr0 = uv_ptrs_starts[i]
            ptrs = uv_ptrs[uv_ptr0 + np.arange(n_terms[i])]
            us = buffer[ptrs[:, None] + np.arange(n_rows[i])[None, :]]
            vs = buffer[
                ptrs[:, None] + np.arange(n_rows[i], n_rows[i] + n_cols[i])[None, :]
            ]
            appxs.append((us.T, vs))
    return appxs
