#! /usr/bin/env python
"""Generate a cropped mesh from a point cloud using pymeshlab and the screened poisson method.

Usage: xyz2ply.py cloud.xyz mesh.ply"""

__author__ = "Benjamin Barad"
__email__ = "benjamin.barad@gmail.com"
__license__ = "GPLv3"

import math

import click
import pymeshlab as pm


@click.command()
@click.argument('xyzfile', type=str)
@click.argument('plyfile', type=str)
@click.option('--pointweight', type=float, default=1)
@click.option('--simplify', type=bool, default=False)
@click.option('--num_faces', type=int, default=300000)
@click.option('--k_neighbors', type=int, default=40)
@click.option('--deldist', type=float, default=1)
@click.option('--smooth_iter', type=int, default=1)
@click.option('--depth', type=int, default=9)
@click.option('--isotropic_remesh', type=bool, default=True)
@click.option('--target_area', type=float, default=1.0)
def xyz_to_ply_from_CLI(xyzfile, plyfile, pointweight, simplify, num_faces, k_neighbors, deldist, smooth_iter, depth, isotropic_remesh, target_area):
    """Click wrapper for xyz_to_ply"""
    xyz_to_ply(xyzfile, plyfile, pointweight=pointweight, simplify=simplify, num_faces=num_faces, k_neighbors=k_neighbors, deldist=deldist, smooth_iter=smooth_iter, depth=depth, isotropic_remesh=isotropic_remesh, target_area=target_area)


def xyz_to_ply(xyzfile, plyfile, pointweight=0.1, simplify=False, num_faces=150000, k_neighbors=500, deldist=1.5, smooth_iter=1, depth=9, isotropic_remesh=True, target_area=1.0):
    """Convert an xyz file to a ply file using pymeshlab

    Arguments:
    xyzfile {str} -- Input xyz filename
    plyfile {str} -- Output ply filename
    simplify {bool} -- If True, will simplify the mesh to a set number of faces. Default False.
    num_faces {int} -- Maximal number of allowed faces after decimation. Default 150000, use more for finer sampling but with greater computational cost.
    depth {int} -- Depth of screened poisson octree. Higher numbers mean more complex meshes but can incorporate stepping artifacts. Default 9.
    pointweight {float} -- Screening weight (0 for max smoothness, 1 to 4 for beter fit to points). Default 0.7.
    k_neighbors {int} -- Number of neighbors for point cloud normal estimation - default 500
    deldist {int} -- Max distance to extrapolate. Default 1.5; distances are in the point cloud distance unit (default nm).
    smooth_iter {int} -- Number of smoothing iterations. Default 1.
    isotropic_remesh {bool} -- If True, will apply isotropic remeshing to create near-equilateral triangles with target average area. Default True.
    target_area {float} -- Target average triangle area in square units (e.g., nm^2). Default 1.0.
    """
    print(f"Processing {xyzfile} into {plyfile}")
    ms = pm.MeshSet()
    ms.load_new_mesh(xyzfile)
    ms.compute_normal_for_point_clouds(k=k_neighbors, smoothiter=smooth_iter) # Predict smooth normals
    ms.generate_surface_reconstruction_screened_poisson(depth=depth, pointweight=pointweight, samplespernode=5.,iters=10, scale=1.2, threads=1) # Screened Poisson

    if ms.current_mesh().face_number() == 0:
        print(f"[ERROR] Screened Poisson reconstruction produced no faces for {xyzfile} - skipping. This can happen when the voxel size is not set correctly in the input MRC file.")
        ms.clear()
        return 1
    
    ms.compute_scalar_by_distance_from_another_mesh_per_vertex(measuremesh=1, refmesh=0, maxdist=pm.PureValue(2 * deldist), signeddist=False) # Delete points that are too far from the reference mesh
    ms.compute_selection_by_condition_per_vertex(condselect = f'(q>={deldist})') # Select vertices beyond the extrapolation distance
    ms.compute_selection_by_condition_per_face(condselect = f'(q0>={deldist} || q1>={deldist} || q2>={deldist})') # Select faces with any vertex beyond the extrapolation distance
    ms.meshing_remove_selected_vertices_and_faces()
    if isotropic_remesh:
        # Calculate target edge length from target area for equilateral triangles
        # Area of equilateral triangle = (sqrt(3)/4) * edge^2
        # So edge = sqrt(4 * area / sqrt(3))
        target_edge_length = math.sqrt(4.0 * target_area / math.sqrt(3.0))
        print(f"Applying isotropic remeshing with target edge length {target_edge_length:.4f} (target area {target_area})")
        ms.meshing_isotropic_explicit_remeshing(targetlen=pm.PureValue(target_edge_length), iterations=3)
    if simplify:
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=num_faces, qualitythr=0.6, preserveboundary=True, preservenormal=True, optimalplacement=True, planarquadric=True) # Simplify
    ms.save_current_mesh(plyfile)
    ms.clear()
    return 0
    

if __name__ == "__main__":
    xyz_to_ply_from_CLI()

