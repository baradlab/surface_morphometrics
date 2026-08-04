"""Tests for the normal-orientation logic (pure numpy; no graph-tool/pycurv needed)."""
import numpy as np
import pytest

from surface_morphometrics import flip_normals as fn


def _curvature_properties(k1, k2):
    """Build the pycurv property set derived from principal curvatures k1 >= k2."""
    return {
        "kappa_1": np.array(k1, dtype=float),
        "kappa_2": np.array(k2, dtype=float),
        "mean_curvature_VV": (np.array(k1, float) + np.array(k2, float)) / 2,
        "gauss_curvature_VV": np.array(k1, float) * np.array(k2, float),
        "curvedness_VV": np.sqrt((np.array(k1, float) ** 2 + np.array(k2, float) ** 2) / 2),
        "shape_index_VV": (2 / np.pi) * np.arctan(
            (np.array(k1, float) + np.array(k2, float))
            / (np.array(k1, float) - np.array(k2, float))),
    }


def test_principal_curvatures_swap_and_negate():
    props = _curvature_properties([0.5, 0.2], [-0.1, -0.3])
    fn.flip_properties(props, [True, True])
    # kappa_1' = -kappa_2, kappa_2' = -kappa_1
    assert np.allclose(props["kappa_1"], [0.1, 0.3])
    assert np.allclose(props["kappa_2"], [-0.5, -0.2])


def test_flip_preserves_the_curvature_identities():
    # The real correctness property: after flipping, every derived quantity must
    # still be the correct function of the (new) principal curvatures.
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=64), rng.normal(size=64)
    k1, k2 = np.maximum(a, b), np.minimum(a, b)
    props = _curvature_properties(k1, k2)
    before = {k: v.copy() for k, v in props.items()}

    fn.flip_properties(props, np.ones(64, dtype=bool))

    new_k1, new_k2 = props["kappa_1"], props["kappa_2"]
    assert np.all(new_k1 >= new_k2)                                    # ordering held
    assert np.allclose(props["mean_curvature_VV"], (new_k1 + new_k2) / 2)
    assert np.allclose(props["gauss_curvature_VV"], new_k1 * new_k2)
    assert np.allclose(props["curvedness_VV"], np.sqrt((new_k1**2 + new_k2**2) / 2))
    assert np.allclose(props["shape_index_VV"],
                       (2 / np.pi) * np.arctan((new_k1 + new_k2) / (new_k1 - new_k2)))

    # ... and the even quantities are untouched while the odd ones negate.
    assert np.allclose(props["gauss_curvature_VV"], before["gauss_curvature_VV"])
    assert np.allclose(props["curvedness_VV"], before["curvedness_VV"])
    assert np.allclose(props["mean_curvature_VV"], -before["mean_curvature_VV"])
    assert np.allclose(props["shape_index_VV"], -before["shape_index_VV"])


def test_flipping_twice_is_the_identity():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=32), rng.normal(size=32)
    props = _curvature_properties(np.maximum(a, b), np.minimum(a, b))
    props["n_v"] = rng.normal(size=(32, 3))
    props["t_1"] = rng.normal(size=(32, 3))
    props["t_2"] = rng.normal(size=(32, 3))
    before = {k: v.copy() for k, v in props.items()}

    mask = np.ones(32, dtype=bool)
    fn.flip_properties(props, mask)
    fn.flip_properties(props, mask)

    for name, original in before.items():
        assert np.allclose(props[name], original), f"{name} did not round-trip"


def test_only_masked_triangles_change():
    props = _curvature_properties([0.5, 0.2], [-0.1, -0.3])
    props["n_v"] = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    fn.flip_properties(props, [True, False])
    assert np.allclose(props["n_v"][1], [0.0, 1.0, 0.0])      # untouched row
    assert np.allclose(props["n_v"][0], [0.0, 0.0, -1.0])     # flipped row
    assert props["kappa_1"][1] == 0.2 and props["kappa_2"][1] == -0.3


def test_normals_negate_and_principal_directions_swap():
    props = {
        "n_v": np.array([[0.0, 0.0, 1.0]]),
        "normal": np.array([[0.0, 0.0, 1.0]]),
        "avg_normals": np.array([[0.0, 0.0, 1.0]]),
        "t_1": np.array([[1.0, 0.0, 0.0]]),
        "t_2": np.array([[0.0, 1.0, 0.0]]),
    }
    fn.flip_properties(props, [True])
    for name in ("n_v", "normal", "avg_normals"):
        assert np.allclose(props[name], [[0.0, 0.0, -1.0]])
    assert np.allclose(props["t_1"], [[0.0, 1.0, 0.0]])
    assert np.allclose(props["t_2"], [[1.0, 0.0, 0.0]])


def test_component_mean_curvatures_are_area_weighted():
    components = np.array([1, 1, 2])
    # component 1: a large slightly-negative triangle outweighs a tiny positive one
    curvature = np.array([-0.1, 10.0, 0.5])
    area = np.array([100.0, 0.01, 3.0])
    got = fn.component_mean_curvatures(components, curvature, area)
    assert got[1] < 0
    assert got[2] == pytest.approx(0.5)


def test_component_mean_curvatures_ignore_nan():
    got = fn.component_mean_curvatures([1, 1], [np.nan, 2.0], [5.0, 1.0])
    assert got[1] == pytest.approx(2.0)


def test_decide_flips_flips_negative_components_only():
    flips = fn.decide_flips({1: -0.05, 2: 0.05, 3: -0.01})
    assert flips == {1: True, 2: False, 3: True}


def test_decide_flips_honors_excluded_labels():
    flips = fn.decide_flips({1: -0.05, 2: -0.05}, exclude_labels=[2])
    assert flips == {1: True, 2: False}


@pytest.mark.parametrize("value,surface,expected", [
    ("3,7", "TE1_IMM.AVV_rh9", {3, 7}),
    ("3 7", "TE1_IMM.AVV_rh9", {3, 7}),
    ("TE1_IMM:3,7", "TE1_IMM.AVV_rh9", {3, 7}),
    ("TE1_OMM:3,7", "TE1_IMM.AVV_rh9", set()),      # different surface
    ("", "TE1_IMM.AVV_rh9", set()),
])
def test_parse_exclude_labels(value, surface, expected):
    assert fn.parse_exclude_labels([value], surface) == expected


def test_parse_exclude_labels_rejects_non_integers():
    import click
    with pytest.raises(click.BadParameter):
        fn.parse_exclude_labels(["3,banana"], "TE1_IMM")


def test_every_pycurv_property_is_classified():
    # The full vertex-property set of a real AVV graph. A pycurv property that is
    # sign-sensitive but unclassified would be left inconsistent with the flipped
    # normals, so this pins the list.
    avv_properties = [
        "area", "avg_normals", "curvedness_VV", "gauss_curvature", "gauss_curvature_VV",
        "kappa_1", "kappa_2", "max_curvature", "mean_curvature", "mean_curvature_VV",
        "min_curvature", "n_v", "normal", "orientation_class", "points",
        "shape_index_VV", "shape_index_cat", "t_1", "t_2", "t_v", "xyz",
    ]
    _, unknown = fn.classify_properties(avv_properties)
    assert unknown == []


def test_unrecognized_property_is_reported():
    _, unknown = fn.classify_properties(["kappa_1", "some_new_signed_thing"])
    assert unknown == ["some_new_signed_thing"]


def test_flip_confidence_flags_balanced_components():
    # A component with lots of curvature that cancels out (H near 0, curvedness
    # large) is the ambiguous case; one whose curvature all points one way is not.
    conf = fn.flip_confidence({1: 0.0001, 2: 0.05}, {1: 0.05, 2: 0.05})
    assert conf[1] < fn.AMBIGUOUS_CONFIDENCE
    assert conf[2] == pytest.approx(1.0)


def test_flip_confidence_handles_zero_curvedness():
    assert fn.flip_confidence({1: 0.0}, {1: 0.0})[1] == 0.0


def _sphere(n=2000, radius=5.0):
    """Fibonacci sphere -> (points, outward unit normals).

    Evenly distributed, so the sampled centroid sits very close to the true
    centre; a random sample leaves a small offset that shows up as alignment
    slightly below 1.
    """
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    directions = np.column_stack([np.cos(theta) * np.sin(phi),
                                  np.sin(theta) * np.sin(phi),
                                  np.cos(phi)])
    return directions * radius, directions


def test_centroid_alignment_is_plus_one_for_outward_sphere_normals():
    xyz, outward = _sphere()
    got = fn.component_centroid_alignment(np.ones(len(xyz)), xyz, outward, np.ones(len(xyz)))
    assert got[1] == pytest.approx(1.0, abs=1e-3)


def test_centroid_alignment_is_minus_one_for_inward_sphere_normals():
    xyz, outward = _sphere()
    got = fn.component_centroid_alignment(np.ones(len(xyz)), xyz, -outward, np.ones(len(xyz)))
    assert got[1] == pytest.approx(-1.0, abs=1e-3)


def test_centroid_alignment_is_near_zero_for_a_flat_sheet():
    # The centroid of an open sheet lies on the sheet, so every normal is
    # perpendicular to the radial direction: the test carries no information.
    grid = np.stack(np.meshgrid(np.linspace(-5, 5, 20), np.linspace(-5, 5, 20)), -1).reshape(-1, 2)
    xyz = np.column_stack([grid, np.zeros(len(grid))])
    normals = np.tile([0.0, 0.0, 1.0], (len(xyz), 1))
    got = fn.component_centroid_alignment(np.ones(len(xyz)), xyz, normals, np.ones(len(xyz)))
    assert abs(got[1]) < 1e-9


def test_centroid_alignment_separates_components():
    xyz_a, out_a = _sphere(n=500, radius=3.0)
    xyz_b, out_b = _sphere(n=500, radius=3.0)
    xyz = np.vstack([xyz_a, xyz_b + 100.0])          # far apart
    normals = np.vstack([out_a, -out_b])             # second one points inward
    components = np.array([1] * 500 + [2] * 500)
    got = fn.component_centroid_alignment(components, xyz, normals, np.ones(1000))
    assert got[1] == pytest.approx(1.0, abs=1e-2)
    assert got[2] == pytest.approx(-1.0, abs=1e-2)


def test_decide_flips_works_on_either_criterion():
    # decide_flips is criterion-agnostic: it flips whatever scores negative.
    assert fn.decide_flips({1: -0.4, 2: 0.9}) == {1: True, 2: False}


def test_manual_flip_rejects_non_integer_labels(tmp_path):
    from click.testing import CliRunner
    graph = tmp_path / "x_oriented.gt"
    graph.write_bytes(b"")
    result = CliRunner().invoke(fn.manual_flip_cli, [str(graph), "--labels", "3,banana"])
    assert result.exit_code != 0
    assert "integer component ids" in result.output


def test_manual_flip_requires_at_least_one_label(tmp_path):
    from click.testing import CliRunner
    graph = tmp_path / "x_oriented.gt"
    graph.write_bytes(b"")
    result = CliRunner().invoke(fn.manual_flip_cli, [str(graph), "--labels", " "])
    assert result.exit_code != 0
    assert "at least one component id" in result.output


def test_manual_flip_toggle_semantics():
    # normals_flipped records orientation relative to the ORIGINAL mesh, so a manual
    # flip XORs it: a component the auto pass flipped and the user flips back reads 0.
    already = np.array([1, 1, 0, 0], dtype=bool)
    manual = np.array([1, 0, 1, 0], dtype=bool)
    assert list((already ^ manual).astype(int)) == [0, 1, 1, 0]


def test_outwardness_is_negated_mean_curvature():
    # pycurv's convention, measured on a meshed sphere of radius 24.99 nm whose
    # normals all point outward (n . radial = +1.000): mean_curvature_VV = -0.0400,
    # exactly -1/R. So outward is NEGATIVE mean curvature.
    assert fn.outwardness_from_curvature({1: -0.04, 2: 0.04}) == {1: 0.04, 2: -0.04}


def test_convex_surface_with_outward_normals_is_left_alone():
    # The sphere case: H = -1/R with normals already outward -> nothing to do.
    assert fn.decide_flips(fn.outwardness_from_curvature({1: -0.04})) == {1: False}


def test_convex_surface_with_inward_normals_is_flipped():
    assert fn.decide_flips(fn.outwardness_from_curvature({1: 0.04})) == {1: True}


def test_curvature_and_centroid_criteria_agree_on_a_sphere():
    # Both criteria must call the same orientation "outward"; they disagreed under
    # the old inverted sign, which made the cross-check report false conflicts.
    outward_curvature = fn.outwardness_from_curvature({1: -0.04})   # normals outward
    centroid_alignment = {1: 1.0}                                   # normals outward
    assert (outward_curvature[1] > 0) == (centroid_alignment[1] > 0)


def test_reverse_winding_only_touches_flipped_triangles():
    pytest.importorskip("graph_tool")
    from graph_tool import Graph

    graph = Graph(directed=False)
    graph.add_vertex(2)
    corners = graph.new_vertex_property("python::object")
    triangle = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    corners[graph.vertex(0)] = list(triangle)
    corners[graph.vertex(1)] = list(triangle)
    graph.vp["points"] = corners

    fn._reverse_winding(graph, np.array([True, False]))

    assert list(graph.vp["points"][graph.vertex(0)]) == triangle[::-1]
    assert list(graph.vp["points"][graph.vertex(1)]) == triangle


def _triangle_polydata(normal=(0.0, 0.0, 1.0), name="n_v"):
    """Two triangles sharing an edge, carrying a per-cell normal array."""
    vtk = pytest.importorskip("vtk")
    from vtk.util.numpy_support import numpy_to_vtk

    points = vtk.vtkPoints()
    for xyz in ([0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]):
        points.InsertNextPoint(*xyz)
    polys = vtk.vtkCellArray()
    for corners in ((0, 1, 2), (1, 3, 2)):
        triangle = vtk.vtkTriangle()
        for slot, corner in enumerate(corners):
            triangle.GetPointIds().SetId(slot, corner)
        polys.InsertNextCell(triangle)
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(polys)
    array = numpy_to_vtk(np.tile(np.asarray(normal, dtype=float), (2, 1)), deep=True)
    array.SetName(name)
    poly.GetCellData().AddArray(array)
    return poly


def test_attach_active_normals_publishes_cell_and_point_normals():
    from vtk.util.numpy_support import vtk_to_numpy
    poly = _triangle_polydata(normal=(0.0, 0.0, 1.0))
    assert poly.GetCellData().GetNormals() is None      # pycurv leaves none active

    assert fn.attach_active_normals(poly) is True

    assert poly.GetCellData().GetNormals().GetName() == "n_v"
    point_normals = vtk_to_numpy(poly.GetPointData().GetNormals())
    assert np.allclose(point_normals, [0.0, 0.0, 1.0])
    assert np.allclose(np.linalg.norm(point_normals, axis=1), 1.0)


def test_attach_active_normals_follows_a_flipped_normal():
    from vtk.util.numpy_support import vtk_to_numpy
    poly = _triangle_polydata(normal=(0.0, 0.0, -1.0))
    fn.attach_active_normals(poly)
    assert np.allclose(vtk_to_numpy(poly.GetPointData().GetNormals()), [0.0, 0.0, -1.0])


def test_attach_active_normals_reports_a_surface_without_n_v():
    poly = _triangle_polydata(name="something_else")
    assert fn.attach_active_normals(poly) is False
    assert poly.GetPointData().GetNormals() is None
