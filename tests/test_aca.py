import time

import numpy as np
import py_aca
import pytest
from test_tde import setup_matrix_test

import cutde.fullspace as FS
import cutde.halfspace as HS
from cutde.aca import call_clu_aca


@pytest.mark.slow
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("F_ordered", [True, False])
@pytest.mark.parametrize("field", ["disp", "strain"])
@pytest.mark.parametrize("module_name", ["HS", "FS"])
def test_aca_slow(dtype, F_ordered, field, module_name):
    # The fast version of this test doesn't test multiple blocks and would miss
    # some potential errors.
    runner(dtype, F_ordered, field, module_name, 10, compare_against_py=True)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("F_ordered", [True, False])
@pytest.mark.parametrize("field", ["disp", "strain"])
@pytest.mark.parametrize("module_name", ["HS"])
def test_aca_fast(dtype, F_ordered, field, module_name):
    runner(dtype, F_ordered, field, module_name, 1, compare_against_py=False)


def runner(
    dtype,
    F_ordered,
    field,
    module_name,
    n_sets,
    compare_against_py=False,
    benchmark_iters=1,
):
    """
    This checks that the OpenCL/CUDA ACA implementation is producing *exactly*
    the same results as the prototype Python implementation and that both ACA
    implementations are within the expected Frobenius norm tolerance of the
    exact calculation.
    """
    module = HS if module_name == "HS" else FS
    if field == "disp":
        matrix_fnc = module.disp_matrix
        aca_fnc = module.disp_aca
        field_spec = module.DISP_SPEC
        vec_dim = 3
    else:
        matrix_fnc = module.strain_matrix
        aca_fnc = module.strain_aca
        field_spec = module.STRAIN_SPEC
        vec_dim = 6

    pts = []
    tris = []
    set_sizes = np.arange(50, 50 + n_sets)
    set_sizes_edges = np.zeros(set_sizes.shape[0] + 1, dtype=np.int32)
    set_sizes_edges[1:] = np.cumsum(set_sizes)
    for i in range(n_sets):
        S = set_sizes[i]
        this_pts, this_tris, _ = setup_matrix_test(
            dtype, F_ordered, n_obs=S, n_src=S, seed=i
        )
        this_pts[:, 0] -= 150
        pts.append(this_pts)
        tris.append(this_tris)
    pts = np.concatenate(pts)
    tris = np.concatenate(tris)

    obs_starts = []
    obs_ends = []
    src_starts = []
    src_ends = []
    for i in range(n_sets):
        for j in range(n_sets):
            obs_starts.append(set_sizes_edges[i])
            obs_ends.append(set_sizes_edges[i + 1])
            src_starts.append(set_sizes_edges[j])
            src_ends.append(set_sizes_edges[j + 1])

    M1 = matrix_fnc(pts, tris, 0.25)
    M1 = M1.reshape((M1.shape[0] * M1.shape[1], M1.shape[2] * M1.shape[3]))

    times = []
    for i in range(benchmark_iters):
        start = time.time()
        if compare_against_py:
            M2 = call_clu_aca(
                pts,
                tris,
                obs_starts,
                obs_ends,
                src_starts,
                src_ends,
                0.25,
                [1e-4] * len(obs_starts),
                [200] * len(obs_starts),
                field_spec,
                Iref0=np.zeros_like(obs_starts),
                Jref0=np.zeros_like(obs_starts),
            )
        else:
            M2 = aca_fnc(
                pts,
                tris,
                obs_starts,
                obs_ends,
                src_starts,
                src_ends,
                0.25,
                [1e-4] * len(obs_starts),
                [200] * len(obs_starts),
            )
        times.append(time.time() - start)

    print("aca runtime, min=", np.min(times), "  median=", np.median(times))

    for block_idx in range(len(obs_starts)):
        os = obs_starts[block_idx]
        oe = obs_ends[block_idx]
        ss = src_starts[block_idx]
        se = src_ends[block_idx]
        block = M1[(os * vec_dim) : (oe * vec_dim), (3 * ss) : (3 * se)]
        U2, V2 = M2[block_idx]
        if compare_against_py:
            U, V = py_aca.ACA_plus(
                block.shape[0],
                block.shape[1],
                lambda Istart, Iend: block[Istart:Iend, :],
                lambda Jstart, Jend: block[:, Jstart:Jend],
                1e-4,
                verbose=False,
                Iref=0,
                Jref=0,
                vec_dim=3 if field == "disp" else 6,
            )

            if pts.dtype.type is np.float64:
                U2, V2 = M2[block_idx]
                np.testing.assert_almost_equal(U, U2, 9)
                np.testing.assert_almost_equal(V, V2, 9)

        diff = block - U2.dot(V2)
        diff_frob = np.sqrt(np.sum(diff**2))
        assert diff_frob < 1e-3


@pytest.mark.parametrize("field_spec", [FS.DISP_SPEC, FS.STRAIN_SPEC])
def test_aca_max_iter_zero(field_spec):
    """When max_iter=0, no ACA terms should be produced and U @ V should be
    the zero matrix."""
    _, vec_dim = field_spec
    n_obs, n_src = 5, 5
    pts, tris, _ = setup_matrix_test(np.float64, False, n_obs=n_obs, n_src=n_src)

    result = call_clu_aca(
        pts,
        tris,
        [0],
        [n_obs],
        [0],
        [n_src],
        0.25,
        [1e-4],
        [0],
        field_spec,
    )

    U, V = result[0]
    assert U.shape == (n_obs * vec_dim, 0), f"Expected 0 terms, got U.shape={U.shape}"
    assert V.shape == (0, n_src * 3), f"Expected 0 terms, got V.shape={V.shape}"
    np.testing.assert_array_equal(U @ V, np.zeros((n_obs * vec_dim, n_src * 3)))


def test_aca_strain_asymmetric_block():
    """Regression test for fworkspace_per_block bug (gh-52).

    With vec_dim=6 (strain) and n_src > 2*n_obs, the workspace was
    undersized causing buffer overflow into adjacent blocks' memory."""
    n_obs, n_src = 5, 20
    pts, tris, _ = setup_matrix_test(np.float64, False, n_obs=n_obs, n_src=n_src)
    pts[:, 0] -= 150

    M = FS.strain_matrix(pts, tris, 0.25)
    M = M.reshape((M.shape[0] * M.shape[1], M.shape[2] * M.shape[3]))

    obs_starts = [0, 0]
    obs_ends = [n_obs, n_obs]
    src_starts = [0, 0]
    src_ends = [n_src, n_src]

    result = call_clu_aca(
        pts,
        tris,
        obs_starts,
        obs_ends,
        src_starts,
        src_ends,
        0.25,
        [1e-4] * 2,
        [200] * 2,
        FS.STRAIN_SPEC,
        Iref0=np.zeros(2, dtype=np.int32),
        Jref0=np.zeros(2, dtype=np.int32),
    )

    for i in range(2):
        U, V = result[i]
        diff = M - U @ V
        diff_frob = np.sqrt(np.sum(diff**2))
        assert (
            diff_frob < 1e-3
        ), f"Block {i}: Frobenius error {diff_frob:.3e} exceeds tolerance"


if __name__ == "__main__":
    runner(np.float32, False, "disp", 40, compare_against_py=False, benchmark_iters=2)
