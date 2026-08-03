#! /usr/bin/env python
"""Step two of the pipeline for surface morphometrics
Take a VTP style mesh generatead through the screen and perform curvature analysis using AVV.

Citation: Barad BA, Medina M et al. Quantifying organellar ultrastructure in cryo-electron tomography using a surface morphometrics pipeline. J Cell Biol 2023.
Pycurv Citation: Salfer M et al. Reliable estimation of membrane curvature for cryo-electron tomography. PLOS Computational Biology 2020.

Two usage options:
1. Run pycurv on a single mesh:
  morphometrics pycurv config.yml mesh.surface.vtp
2. Run pycurv on all meshes in the working directory:
  morphometrics pycurv config.yml
"""

__author__ = "Benjamin Barad"
__email__ = "benjamin.barad@gmail.com"
__license__ = "GPLv3"

# Set OMP_NUM_THREADS=1 before any imports to prevent OpenMP + fork deadlocks
# when graph-tool is used with multiprocessing
import os
os.environ["OMP_NUM_THREADS"] = "1"

import glob
import sys

import click
import yaml

from .config_utils import load_config

from . import curvature


@click.command(name="pycurv")
@click.argument("configfile", type=click.Path(exists=True))
@click.argument("surface", required=False, default=None)
@click.option("-f", "--force", is_flag=True, default=False,
              help="Skip interactive confirmation prompts.")
@click.option("--gpu", is_flag=True,
              help="Use pycurv-gpu (CUDA) instead of CPU pycurv.")
def run_pycurv_cli(configfile, surface, force, gpu):
    """Run pycurv vector-voting curvature analysis on surface meshes.

    CONFIGFILE: path to config.yml.
    SURFACE: optional single .surface.vtp to process; if omitted, all surfaces
    in work_dir are processed.
    """
    config = load_config(configfile, require=("work_dir",))
    use_gpu = gpu or config["curvature_measurements"].get("use_gpu", False)
    if use_gpu:
        curvature._require_gpu_backend(config["curvature_measurements"].get("gpu_device"))

    # Warn if configured cores exceed logical cores
    cores = config["cores"]
    logical_cores = os.cpu_count()
    if cores > logical_cores:
        print(f"WARNING: Configured cores ({cores}) exceeds the number of logical cores ({logical_cores}).")
        print("This may cause performance degradation due to oversubscription.")
        if not force:
            answer = input("Continue anyway? [y/n] ")
            if answer != "y":
                sys.exit(1)

    # Figure out what files will be run
    if surface is None:
        print("No input file specified - will run on all VTP files in the working directory")
        mesh_files = glob.glob(config["work_dir"] + "*.surface.vtp")
        mesh_files = [os.path.basename(f) for f in mesh_files]
    else:
        print("Input file specified - will run on this file only")
        mesh_files = [surface]

    # Check that work dir exists for outputs
    if not os.path.isdir(config["work_dir"]):
        os.mkdir(config["work_dir"])

    failed_surfaces = []
    for i, surface_file in enumerate(mesh_files):
        print("Processing {} ({}/{})".format(surface_file, i + 1, len(mesh_files)))
        try:
            curvature.run_pycurv(surface_file, config["work_dir"],
                                 scale=1.0,
                                 radius_hit=config["curvature_measurements"]["radius_hit"],
                                 min_component=config["curvature_measurements"]["min_component"],
                                 exclude_borders=config["curvature_measurements"]["exclude_borders"],
                                 cores=config["cores"],
                                 use_gpu=use_gpu,
                                 gpu_device=config["curvature_measurements"].get("gpu_device"),
                                 gpu_batch_size=config["curvature_measurements"].get("gpu_batch_size", 1024))
            print("Completed {}\n".format(surface_file))
        except Exception as e:
            print("WARNING: Skipping {} due to error: {}\n".format(surface_file, e))
            failed_surfaces.append(surface_file)

    if failed_surfaces:
        print("The following surfaces failed and were skipped:")
        for s in failed_surfaces:
            print("  - {}".format(s))

    print("-------------------------------------------------------")
    label = "GPU pycurv" if use_gpu else "Pycurv"
    print(f"{label} complete. It is highly recommended to check the AVV vtp file with paraview to confirm good results.")
    print("Pycurv Citation: Salfer M, Collado JF, Baumeister W, Fernández-Busnadiego R, Martínez-Sánchez A. Reliable estimation of membrane curvature for cryo-electron tomography. PLOS Comp Biol 2020.")
    print("Pipeline Citation: Barad BA*, Medina M*, Fuentes D, Wiseman RL, Grotjahn DA. Quantifying organellar ultrastructure in cryo-electron tomography using a surface morphometrics pipeline. J Cell Biol 2023.")


if __name__ == "__main__":
    run_pycurv_cli()
