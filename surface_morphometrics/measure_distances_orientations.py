#! /usr/bin/env python
"""Step three of the pipeline

Take a series of triange graphs in gt format from pycurv and measure
distances within and between surfaces as well as orientations relative to
the growth plane and to each other.

Two usage options:
1. Measure distances and orientations on a single **tomogram** worth of graphs - yes, supply a segmentation file, not the meshes.
  morphometrics distances_orientations config.yml segmentation.mrc
2. Measure distances and orientations for the graphs for every tomogram in a folder:
  morphometrics distances_orientations config.yml

Since these distance measurements can take a minute or two per measurement per surface,
it may be worth doing this in parallel using option 1."""

__author__ = "Benjamin Barad"
__email__ = "benjamin.barad@gmail.com"
__license__ = "GPLv3"

import os
import glob

import click
import yaml

from .config_utils import load_config, resolve_distance_targets

from . import intradistance_verticality
from . import interdistance_orientation


@click.command(name="distances_orientations")
@click.argument("configfile", type=click.Path(exists=True))
@click.argument("segmentation", required=False, default=None)
@click.option("-f", "--force", is_flag=True, default=False, hidden=True,
              help="Deprecated no-op: distances_orientations no longer prompts.")
def distances_orientations_cli(configfile, segmentation, force):
    """Measure intra- and inter-surface distances and orientations.

    CONFIGFILE: path to config.yml.
    SEGMENTATION: optional single segmentation .mrc whose graphs to process; if
    omitted, all segmentations in seg_dir are processed.
    """
    config = load_config(configfile, require=("seg_dir", "work_dir", "segmentation_values"))

    if force:
        print("NOTE: -f/--force is no longer needed (distances_orientations does not prompt) "
              "and will be removed in 2.0; drop it from your scripts.")
    # See if a specific file was specified
    if segmentation is None:
        print("No input file specified - will run on meshes for all segmentation files in the data directory")
        print("Pattern Matched: " + config["seg_dir"] + "*.mrc")
        segmentation_files = glob.glob(config["seg_dir"] + "*.mrc")
        segmentation_files = [os.path.basename(f) for f in segmentation_files]
        print(segmentation_files)
    else:
        print("Input file specified - will run on meshes associated with this segmentation only")
        segmentation_files = [segmentation]

    dist_settings = config["distance_and_orientation_measurements"]

    # intra/inter default to all-vs-all when omitted (every surface, every pair);
    # an explicit empty (intra: [] / inter: {}) opts out. See resolve_distance_targets.
    labels = list(config["segmentation_values"].keys())
    intra, inter, intra_defaulted, inter_defaulted = resolve_distance_targets(dist_settings, labels)
    if intra_defaulted:
        print(f"No 'intra' configured - defaulting to all surfaces: {intra}")
        print("  (set 'intra: []' under distance_and_orientation_measurements to skip)")
    if inter_defaulted:
        print("No 'inter' configured - defaulting to all surface pairs.")
        print("  (set 'inter: {}' under distance_and_orientation_measurements to skip)")

    print("Distance and orientation settings:")
    print("Will measure intra-surface distances for:")
    if intra:
        print(intra)
        if dist_settings["verticality"]:
            print("Will also measure verticality for those surfaces")
    if inter:
        print("Will make (bi-directional) inter-surface distance measurements for:")
        for surface, comparison in inter.items():
            print(surface + f": {comparison}")
        if dist_settings["relative_orientation"]:
            print("Will also measure relative orientation for those surfaces")
    print("---------")

    radius_hit = config["curvature_measurements"]["radius_hit"]
    for file in segmentation_files:
        file_base = file[:-4]
        print(file_base)
        # Intra-surface distances
        if intra:
            for label in intra:
                graphname = config["work_dir"] + file_base + "_" + label + ".AVV_rh" + str(radius_hit) + ".gt"
                if not os.path.isfile(graphname):
                    print("No file found for " + graphname)
                    print("Skipping this label for this tomogram")
                    continue
                print("Intra-surface distances for " + graphname)
                surfacename = graphname[:-3] + ".vtp"
                if dist_settings["verticality"]:
                    intradistance_verticality.surface_verticality(graphname)
                intradistance_verticality.surface_self_distances(graphname, surfacename,
                                                                 dist_min=dist_settings["mindist"],
                                                                 dist_max=dist_settings["maxdist"],
                                                                 tolerance=dist_settings["tolerance"],
                                                                 exportcsv=True)
        # Inter-surface distances
        if inter:
            for label1, comparison in inter.items():
                graphname1 = config["work_dir"] + file_base + "_" + label1 + ".AVV_rh" + str(radius_hit) + ".gt"
                if not os.path.isfile(graphname1):
                    print("No file found for " + graphname1)
                    print("Skipping all intersurface measurements for label " + label1)
                    continue
                for label2 in comparison:
                    graphname2 = config["work_dir"] + file_base + "_" + label2 + ".AVV_rh" + str(radius_hit) + ".gt"
                    if not os.path.isfile(graphname2):
                        print("No file found for " + graphname2)
                        print("Skipping comparison with " + label2)
                        continue
                    print("Inter-surface distances for " + label1 + " and " + label2)
                    interdistance_orientation.surface_to_surface(graphname1, label1,
                                                                 graphname2, label2,
                                                                 orientation=dist_settings["relative_orientation"],
                                                                 save_neighbor_index=True,
                                                                 exportcsv=True)

    print("---------")
    print("Done with distance and orientation measurements. Check out the output vtp and csv files and then start getting to work doing stats!")
    print("Citation: Barad BA*, Medina M*, Fuentes D, Wiseman RL, Grotjahn DA. Quantifying organellar ultrastructure in cryo-electron tomography using a surface morphometrics pipeline. J Cell Biol 2023.")


if __name__ == "__main__":
    distances_orientations_cli()
