#! /usr/bin/env python
"""Generate a VTK style vtp file from a ply file

usage: ply2vtp.py mesh.ply mesh.vtp"""

__author__ = "Benjamin Barad"
__email__ = "benjamin.barad@gmail.com"
__license__ = "GPLv3"

import gc

import click


@click.command()
@click.argument('input_ply', type=str)
@click.argument('output_vtp', type=str)
def convert_from_CLI(input_ply, output_vtp):
    """Click wrapper for convert script"""
    ply_to_vtp(input_ply, output_vtp)

def ply_to_vtp(plyfilename, vtpfilename):
    """Convert an input ply file to a vtp file"""
    # vtk is imported here, not at module scope, so that importing this module
    # (and segmentation_to_meshes, which orchestrates it) does not require vtk.
    import vtk

    plyfile = vtk.vtkPLYReader()
    plyfile.SetFileName(plyfilename)
    plyfile.Update()
    surf = plyfile.GetOutput() 
    writer = vtk.vtkXMLPolyDataWriter()
    fname = vtpfilename
    writer.SetFileName(fname)
    writer.SetInputData(surf)
    if writer.Write() != 1:
        raise RuntimeError(f"Error writing the vtp file {fname}.")
    gc.collect()






if __name__ == '__main__':
    convert_from_CLI()