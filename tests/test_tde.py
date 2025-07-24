import time

import numpy as np
import pytest
import scipy.io

import cutde.fullspace as FS
import cutde.halfspace as HS


def get_pt_grid():
    n = 11
    x = np.linspace(-3, 3, n)
    y = np.linspace(-3, 3, n)
    z = np.linspace(-3, 3, n)
    X, Z, Y = np.meshgrid(x, z, y)
    test_pts = np.hstack([arr.flatten()[:, np.newaxis] for arr in [X, Y, Z]])
    return test_pts


def get_simple_test():
    correct = scipy.io.loadmat("tests/result_simple.mat")
    tri = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    slip = np.array([1.0, 0, 0])
    return correct, get_pt_grid(), tri, slip


def get_complex_test():
    correct = scipy.io.loadmat("tests/result_complex.mat")
    tri = np.array([[0, 0.1, 0.1], [1, -0.2, -0.2], [1, 1, 0.3]])
    slip = [1.3, 1.4, 1.5]
    return correct, get_pt_grid(), tri, slip


def get_halfspace_test():
    correct = scipy.io.loadmat("tests/result_halfspace.mat")
    tri = np.array([[0, 0.1, -0.1], [1, -0.2, -0.2], [1, 1, -0.3]])
    slip = [1.3, 1.4, 1.5]
    obs_pts = get_pt_grid()
    obs_pts[:, 2] -= 3
    return correct, obs_pts, tri, slip


def get_halfspace_horizontal_test():
    correct = scipy.io.loadmat("tests/result_halfspace_horizontal.mat")
    tri = np.array([[-1, 0, -1], [1, 0.0, -1], [0, 1.0, -1]])
    slip = [1.0, 0.5, -0.3]
    obs_pts = get_pt_grid()
    obs_pts[:, 2] -= 3
    return correct, obs_pts, tri, slip


def get_fullspace_horizontal_test():
    correct = scipy.io.loadmat("tests/result_fullspace_horizontal.mat")
    tri = np.array([[-1, 0, -1], [1, 0.0, -1], [0, 1.0, -1]])
    slip = [1.0, 0.5, -0.3]
    obs_pts = get_pt_grid()
    return correct, obs_pts, tri, slip


def py_tde_tester(setup_fnc, N_test=-1):
    correct, test_pts, tri, slip = setup_fnc()

    if N_test == -1:
        N_test = correct["UEf"].shape[0]

    results = np.empty((N_test, 3))
    start = time.time()
    for i in range(N_test):
        pt = test_pts[i, :]
        results[i, :] = FS.py_disp(pt, tri, slip, 0.25)
        np.testing.assert_almost_equal(results[i, 0], correct["UEf"][i, 0])
        np.testing.assert_almost_equal(results[i, 1], correct["UNf"][i, 0])
        np.testing.assert_almost_equal(results[i, 2], correct["UVf"][i, 0])
        if i % 10000 == 0 and i != 0:
            print(i, (time.time() - start) / i)

    np.testing.assert_almost_equal(results[:, 0], correct["UEf"][:N_test, 0])
    np.testing.assert_almost_equal(results[:, 1], correct["UNf"][:N_test, 0])
    np.testing.assert_almost_equal(results[:, 2], correct["UVf"][:N_test, 0])


def test_py_simple_reduced():
    py_tde_tester(get_simple_test, N_test=10)


@pytest.mark.slow
def test_py_simple():
    py_tde_tester(get_simple_test, N_test=1000)


def test_py_complex_reduced():
    py_tde_tester(get_complex_test, N_test=10)


@pytest.mark.slow
def test_py_complex():
    py_tde_tester(get_complex_test, N_test=1000)


def cluda_tde_tester(setup_fnc, N_test=None):
    correct, test_pts, tri, slip = setup_fnc()

    if N_test is None:
        N_test = correct["UEf"].shape[0]

    nu = 0.25
    sm = 1.0

    tris = np.array([tri] * N_test, dtype=np.float64)
    slips = np.array([slip] * N_test)

    disp = FS.disp(test_pts[:N_test], tris, slips, nu)
    strain = FS.strain(test_pts[:N_test], tris, slips, nu)
    stress = FS.strain_to_stress(strain, sm, nu)

    np.testing.assert_almost_equal(disp[:, 0], correct["UEf"][:N_test, 0])
    np.testing.assert_almost_equal(disp[:, 1], correct["UNf"][:N_test, 0])
    np.testing.assert_almost_equal(disp[:, 2], correct["UVf"][:N_test, 0])
    np.testing.assert_almost_equal(strain, correct["Strain"][:N_test])
    np.testing.assert_almost_equal(stress, correct["Stress"][:N_test])

    test_ptsF = np.asfortranarray(test_pts[:N_test])
    trisF = np.asfortranarray(tris)
    slipsF = np.asfortranarray(slips)
    dispF = FS.disp(test_ptsF, trisF, slipsF, 0.25)
    np.testing.assert_almost_equal(disp, dispF)


def test_cluda_simple_reduced():
    cluda_tde_tester(get_simple_test, N_test=10)


@pytest.mark.slow
def test_cluda_simple():
    cluda_tde_tester(get_simple_test)


def test_halfspace():
    correct, test_pts, tri, slip = get_halfspace_test()

    N_test = None
    if N_test is None:
        N_test = correct["UEf"].shape[0]

    tris = np.array([tri] * N_test, dtype=np.float64)
    slips = np.array([slip] * N_test)

    nu = 0.25
    disp = HS.disp(test_pts[:N_test], tris, slips, nu)
    strain = HS.strain(test_pts[:N_test], tris, slips, nu)

    np.testing.assert_almost_equal(disp[:, 0], correct["UEf"][:N_test, 0])
    np.testing.assert_almost_equal(disp[:, 1], correct["UNf"][:N_test, 0])
    np.testing.assert_almost_equal(disp[:, 2], correct["UVf"][:N_test, 0])
    np.testing.assert_almost_equal(strain, correct["Strain"][:N_test])


@pytest.mark.parametrize("module_name", ["HS", "FS"])
def test_horizontal(module_name):
    if module_name == "HS":
        correct, test_pts, tri, slip = get_halfspace_horizontal_test()
        module = HS
    else:
        correct, test_pts, tri, slip = get_fullspace_horizontal_test()
        module = FS

    N_test = None
    if N_test is None:
        N_test = correct["UEf"].shape[0]

    tris = np.array([tri] * N_test, dtype=np.float64)
    slips = np.array([slip] * N_test)

    nu = 0.25
    disp = module.disp(test_pts[:N_test], tris, slips, nu)
    strain = module.strain(test_pts[:N_test], tris, slips, nu)

    np.testing.assert_almost_equal(disp[:, 0], correct["UEf"][:N_test, 0])
    np.testing.assert_almost_equal(disp[:, 1], correct["UNf"][:N_test, 0])
    np.testing.assert_almost_equal(disp[:, 2], correct["UVf"][:N_test, 0])
    np.testing.assert_almost_equal(strain, correct["Strain"][:N_test])


def setup_matrix_test(dtype, F_ordered, n_obs=10, n_src=10, seed=10):
    rand = np.random.RandomState(seed)

    def random_vals(shape, max_val):
        out = (rand.rand(*shape) * max_val).astype(dtype)
        return np.asfortranarray(out) if F_ordered else out

    pts = random_vals((n_obs, 3), max_val=100)
    tris = random_vals((n_src, 3, 3), max_val=100)
    slips = random_vals((n_src, 3), max_val=100)
    # Push the pts and triangles below the free surface. This doesn't matter
    # for fullspace tests but is very important for the halfspace tests.
    pts[:, 2] -= np.max(pts[:, 2]) + 1
    tris[:, :, 2] -= np.max(tris[:, :, 2]) + 1

    return pts, tris, slips


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("F_ordered", [True, False])
@pytest.mark.parametrize("field", ["disp", "strain"])
@pytest.mark.parametrize("module_name", ["HS", "FS"])
def test_matrix(dtype, F_ordered, field, module_name):
    pts, tris, slips = setup_matrix_test(dtype, F_ordered)

    module = HS if module_name == "HS" else FS
    if field == "disp":
        simple_fnc = module.disp
        matrix_fnc = module.disp_matrix
    else:
        simple_fnc = module.strain
        matrix_fnc = module.strain_matrix

    mat1 = []
    slips[:] = 1
    for i in range(pts.shape[0]):
        tiled_pt = np.tile(pts[i, np.newaxis, :], (tris.shape[0], 1))
        mat1.append(simple_fnc(tiled_pt, tris, slips, 0.25))
    S1 = np.sum(np.array(mat1), axis=1).flatten()

    S2 = matrix_fnc(pts, tris, 0.25).reshape((-1, slips.size)).dot(slips.flatten())

    assert np.isfinite(S2).all()
    if pts.dtype.type in [np.float32, np.int32]:
        rtol, atol = 1e-4, 1e-5
    else:
        rtol, atol = 4e-10, 1e-15
    np.testing.assert_allclose(S1, S2, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("F_ordered", [True, False])
@pytest.mark.parametrize("field", ["disp", "strain"])
@pytest.mark.parametrize("module_name", ["HS", "FS"])
def test_matrix_free(dtype, F_ordered, field, module_name):
    pts, tris, slips = setup_matrix_test(dtype, F_ordered)

    module = HS if module_name == "HS" else FS
    if field == "disp":
        matrix_fnc = module.disp_matrix
        free_fnc = module.disp_free
    else:
        matrix_fnc = module.strain_matrix
        free_fnc = module.strain_free

    S1 = matrix_fnc(pts, tris, 0.25).reshape((-1, slips.size)).dot(slips.flatten())
    S2 = free_fnc(pts, tris, slips, 0.25).flatten()

    assert np.isfinite(S2).all()
    if pts.dtype.type in [np.float32, np.int32]:
        rtol, atol = 1e-4, 3e-5
    else:
        rtol, atol = 4e-10, 1e-15
    np.testing.assert_allclose(S1, S2, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("F_ordered", [True, False])
@pytest.mark.parametrize("field", ["disp", "strain"])
@pytest.mark.parametrize("module_name", ["HS", "FS"])
def test_block(dtype, F_ordered, field, module_name):
    pts, tris, slips = setup_matrix_test(dtype, F_ordered)

    module = HS if module_name == "HS" else FS
    if field == "disp":
        matrix_fnc = module.disp_matrix
        block_fnc = module.disp_block
    else:
        matrix_fnc = module.strain_matrix
        block_fnc = module.strain_block

    M1 = matrix_fnc(pts, tris, 0.25)

    src_start = []
    src_end = []
    obs_start = []
    obs_end = []
    # Break apart the full matrix into a bunch of non-overlapping chunks to
    # test the blocking
    obs_edges = [0, 4, 5, pts.shape[0]]
    src_edges = [0, 1, 3, 7, tris.shape[0]]
    for i in range(len(obs_edges) - 1):
        for j in range(len(src_edges) - 1):
            obs_start.append(obs_edges[i])
            obs_end.append(obs_edges[i + 1])
            src_start.append(src_edges[j])
            src_end.append(src_edges[j + 1])
    block_mat, block_start = block_fnc(
        pts, tris, obs_start, obs_end, src_start, src_end, 0.25
    )
    assert block_start.shape[0] == len(src_start) + 1
    assert block_mat.size == M1.size

    # Piece the blocks back together.
    M2 = np.empty_like(M1)
    for i in range(len(src_start)):
        os = obs_start[i]
        oe = obs_end[i]
        ss = src_start[i]
        se = src_end[i]
        M2[os:oe, :, ss:se, :] = block_mat[block_start[i] : block_start[i + 1]].reshape(
            (oe - os, M1.shape[1], se - ss, 3)
        )

    tol = 1e-5 if pts.dtype.type in [np.float32, np.int32] else 1e-10
    np.testing.assert_allclose(M1, M2, rtol=tol, atol=tol)


# Collection of degenerate triangles
DEGENERATE_TRIANGLES = [
    np.array(
        [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    ),  # All vertices coincide
    np.array(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]
    ),  # Collinear vertices
    np.array(
        [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
    ),  # Duplicate vertices
    1e-20
    * np.array(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
    ),  # Nearly degenerate
]


@pytest.mark.parametrize("triangles", DEGENERATE_TRIANGLES)
@pytest.mark.parametrize("module", [FS, HS])
def test_degenerate_triangles_no_nan(module, triangles):
    """
    Verify that degenerate triangles do not produce NaN values.
    """
    obs_pts = np.array([[0.73, -0.41, -1.2]])
    slip = np.array([[1.3, 0.7, 0.3]])
    disp = module.disp(obs_pts, triangles, slip, 0.25)
    strain = module.strain(obs_pts, triangles, slip, 0.25)
    assert np.all(np.isfinite(disp))
    assert np.all(np.isfinite(strain))


@pytest.mark.parametrize("triangles", DEGENERATE_TRIANGLES)
@pytest.mark.parametrize("module", [FS, HS])
def test_degenerate_triangles_zero_result(module, triangles):
    """
    Verify that degenerate triangles produce a zero result.
    """
    obs_pts = np.array([[0.73, -0.41, -1.2]])
    slip = np.array([[1.3, 0.7, 0.3]])
    disp = module.disp(obs_pts, triangles, slip, 0.25)
    strain = module.strain(obs_pts, triangles, slip, 0.25)
    assert np.allclose(disp, 0.0)
    assert np.allclose(strain, 0.0)


@pytest.mark.parametrize("module", [FS, HS])
def test_mixed_degenerate_nondegenerate_triangles(module):
    """
    Verify that calculations work correctly when degenerate and non-degenerate
    triangles are mixed in a single computation. This tests the robustness of
    the degeneracy checks in realistic scenarios.
    """
    # Mix of triangles: degenerate and non-degenerate
    mixed_triangles = np.array(
        [
            # Non-degenerate triangle
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            # Degenerate: all vertices coincide
            [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.5, 0.5, 0.0]],
            # Non-degenerate triangle
            [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0]],
            # Degenerate: collinear vertices
            [[1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [3.0, 3.0, 0.0]],
            # Non-degenerate triangle
            [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.5, 1.0, 0.0]],
        ]
    )

    obs_pts = np.array(
        [
            [0.5, 0.5, -1.0],  # For triangle 0
            [1.5, 0.2, 0.5],  # For triangle 1
            [2.5, 0.5, -0.5],  # For triangle 2
            [2.0, 2.5, 1.0],  # For triangle 3
            [-0.5, 0.5, 0.5],  # For triangle 4
        ]
    )
    slip = np.array(
        [
            [1.0, 0.0, 0.0],  # Non-degenerate
            [0.5, 0.5, 0.5],  # Degenerate (should have no effect)
            [0.0, 1.0, 0.0],  # Non-degenerate
            [1.0, 1.0, 1.0],  # Degenerate (should have no effect)
            [0.0, 0.0, 1.0],  # Non-degenerate
        ]
    )

    # Calculate displacement and strain
    disp = module.disp(obs_pts, mixed_triangles, slip, 0.25)
    strain = module.strain(obs_pts, mixed_triangles, slip, 0.25)

    # Verify no NaN or infinite values
    assert np.all(np.isfinite(disp))
    assert np.all(np.isfinite(strain))

    # Calculate contributions from only non-degenerate triangles
    non_degenerate_indices = [0, 2, 4]  # Indices of non-degenerate triangles
    non_degenerate_triangles = mixed_triangles[non_degenerate_indices]
    non_degenerate_slip = slip[non_degenerate_indices]
    non_degenerate_obs_pts = obs_pts[non_degenerate_indices]

    disp_nondegenerate = module.disp(
        non_degenerate_obs_pts, non_degenerate_triangles, non_degenerate_slip, 0.25
    )
    strain_nondegenerate = module.strain(
        non_degenerate_obs_pts, non_degenerate_triangles, non_degenerate_slip, 0.25
    )

    # Results should be the same (degenerate triangles contribute zero)
    # Compare only the non-degenerate triangle results
    disp_selected = disp[non_degenerate_indices]
    strain_selected = strain[non_degenerate_indices]
    np.testing.assert_allclose(
        disp_selected, disp_nondegenerate, rtol=1e-10, atol=1e-15
    )
    np.testing.assert_allclose(
        strain_selected, strain_nondegenerate, rtol=1e-10, atol=1e-15
    )

    # Verify that degenerate triangles produce zero results
    degenerate_indices = [1, 3]  # Indices of degenerate triangles
    disp_degenerate = disp[degenerate_indices]
    strain_degenerate = strain[degenerate_indices]
    np.testing.assert_allclose(disp_degenerate, 0.0, atol=1e-15)
    np.testing.assert_allclose(strain_degenerate, 0.0, atol=1e-15)
