#! /usr/bin/env python
"""The `morphometrics` command-line interface.

A single grouped command exposes the surface morphometrics pipeline as
subcommands, e.g. `morphometrics make_meshes config.yml`. Subcommands are loaded
lazily: the module backing a subcommand is only imported when that subcommand is
actually invoked. This keeps `morphometrics --help` fast and importable even when
heavy optional dependencies (graph-tool, pymeshlab, pycurv) are not installed.
"""

__author__ = "Benjamin Barad"
__email__ = "benjamin.barad@gmail.com"
__license__ = "GPLv3"

import importlib
import importlib.metadata
import os

import click

# subcommand name -> ("module:click_command_attr", "short help"). Imported on demand.
# The short help is kept here so `morphometrics --help` can list every command
# without importing the (heavy) modules that back them.
_LAZY_COMMANDS = {
    "fetch_example": ("surface_morphometrics.fetch_example:fetch_example_cli",
                      "Download example data for testing (small Zenodo set or full EMPIAR data)."),
    "validate": ("surface_morphometrics.validate:validate_cli",
                 "Check a config and its seg/tomo folders are consistent before running."),
    "status": ("surface_morphometrics.status:status_cli",
               "Summarize what has been computed for each segmentation in the dataset."),
    "make_meshes": ("surface_morphometrics.segmentation_to_meshes:make_meshes_cli",
                    "Convert segmentation MRCs into membrane meshes (step 1)."),
    "pycurv": ("surface_morphometrics.run_pycurv:run_pycurv_cli",
               "Run pycurv curvature analysis on meshes (step 2)."),
    "distances_orientations": ("surface_morphometrics.measure_distances_orientations:distances_orientations_cli",
                               "Measure intra/inter-surface distances and orientations (step 3)."),
    "sample_density": ("surface_morphometrics.sample_density:sample_density_cli",
                       "Sample tomogram density along surface normals (thickness step 1)."),
    "measure_thickness": ("surface_morphometrics.measure_thickness:measure_thickness_cli",
                          "Estimate membrane thickness from sampled density (thickness step 2)."),
    "refine_mesh": ("surface_morphometrics.refine_mesh:refine_mesh_cli",
                    "Density-guided mesh refinement (optional)."),
    "accept_refinement": ("surface_morphometrics.accept_refinement:accept_refinement_cli",
                          "Commit a chosen refinement iteration as the working surface."),
    "stats": ("surface_morphometrics.morphometrics_stats:assemble_experiment_pickle",
              "Assemble an Experiment pickle for statistics."),
    "histogram": ("surface_morphometrics.single_file_histogram:main",
                  "Area-weighted histogram of one feature from one CSV."),
    "hist2d": ("surface_morphometrics.single_file_2d:main",
               "Area-weighted 2D histogram of two features from one CSV."),
    "violin": ("surface_morphometrics.feature_violin:violin_cli",
               "Violin plot of one feature across classes, one point per tomogram."),
    "generate_patches": ("surface_morphometrics.generate_patches:generate_patches_cli",
                         "Place protein-centered membrane patches from a STAR file (+ random controls)."),
    "label_components": ("surface_morphometrics.label_connected_components:label_components_cli",
                         "Label connected components of a graph for per-region stats."),
    "extract_patches": ("surface_morphometrics.extract_patches:extract_patches_cli",
                        "Split a graph into per-region files by label or property range."),
    "patch_statistics": ("surface_morphometrics.patch_statistics:patch_statistics_cli",
                         "Area-weighted per-region (patch/component) statistics from CSVs."),
    "export_obj": ("surface_morphometrics.export_obj:export_obj_cli",
                   "Export a quantified surface to a colormapped OBJ + MTL (Blender etc.)."),
}

# How commands are grouped and ordered in `--help` (instead of alphabetically),
# so the standard pipeline order is obvious. `new_config` is registered eagerly
# but listed here for display.
_SECTIONS = [
    ("Setup", ["new_config", "validate", "fetch_example", "status"]),
    ("Pipeline (run in this order)",
     ["make_meshes", "pycurv", "distances_orientations", "sample_density", "measure_thickness"]),
    ("Optional mesh refinement (after pycurv, before distances)",
     ["refine_mesh", "accept_refinement"]),
    ("Statistics & plotting", ["stats", "histogram", "hist2d", "violin"]),
    ("Patch & region analysis (optional)",
     ["generate_patches", "label_components", "extract_patches", "patch_statistics"]),
    ("Export & visualization", ["export_obj"]),
]

# Printed after a subcommand finishes successfully, to point at the next step.
_NEXT_HINTS = {
    "new_config": "Edit the generated config.yml, then run:  morphometrics make_meshes config.yml",
    "fetch_example": "Next:  cd into the example folder, then run:  morphometrics make_meshes config.yml",
    "validate": "Next:  if validation passed, run:  morphometrics make_meshes config.yml",
    "make_meshes": "Next:  morphometrics pycurv config.yml <name>.surface.vtp   (run per-surface, in parallel on a cluster if you can)",
    "pycurv": "Next:  morphometrics distances_orientations config.yml   (or first refine: morphometrics refine_mesh config.yml)",
    "refine_mesh": "Next:  inspect the convergence plots, then commit one iteration:  morphometrics accept_refinement config.yml <step>",
    "accept_refinement": "Next:  morphometrics distances_orientations config.yml   (re-run morphometrics pycurv first if you accepted a lightweight iteration)",
    "distances_orientations": "Next:  for thickness, morphometrics sample_density config.yml then morphometrics measure_thickness config.yml; otherwise start your stats.",
    "sample_density": "Next:  morphometrics measure_thickness config.yml",
    "measure_thickness": "Next:  assemble results with  morphometrics stats config.yml <name>,  or plot with  morphometrics histogram / hist2d",
    "stats": "Next:  plot features with  morphometrics histogram <file>.csv -n <feature>  or  morphometrics hist2d <file>.csv -n1 <a> -n2 <b>",
    "generate_patches": "Next:  summarize per patch with  morphometrics patch_statistics config.yml  (or split files with  morphometrics extract_patches config.yml --by <prefix>_patch_number)",
    "label_components": "Next:  summarize per component with  morphometrics patch_statistics config.yml --pattern '*_components.csv'",
    "patch_statistics": "Next:  plot the per-region table with  morphometrics histogram / hist2d, or analyze patch_statistics.csv in pandas",
}


# Third-party packages (e.g. the napari GUI) can contribute extra subcommands by
# advertising a "morphometrics.commands" entry point whose target is a click
# Command. That target module must be import-light — do heavy imports (napari/Qt)
# inside the command callback — so `morphometrics --help` stays fast.
_PLUGIN_GROUP = "morphometrics.commands"


def _plugin_entry_points():
    """name -> EntryPoint for every installed `morphometrics.commands` plugin."""
    try:
        return {ep.name: ep for ep in importlib.metadata.entry_points(group=_PLUGIN_GROUP)}
    except Exception:
        return {}


class LazyGroup(click.Group):
    """A click Group that imports a subcommand's module only when invoked.

    `format_commands` is overridden so `--help` lists every subcommand using the
    static short help above, without importing any backing module.
    """

    def list_commands(self, ctx):
        return sorted(
            set(super().list_commands(ctx))
            | set(_LAZY_COMMANDS)
            | set(_plugin_entry_points())
        )

    def get_command(self, ctx, name):
        # Eagerly-registered commands (e.g. new_config) take precedence.
        command = super().get_command(ctx, name)
        if command is not None:
            return command
        target = _LAZY_COMMANDS.get(name)
        if target is not None:
            module_name, attr = target[0].split(":")
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:  # surface a clear message if a heavy dep is missing
                raise click.ClickException(
                    f"Could not load the '{name}' command ({module_name}): {exc}"
                )
            return getattr(module, attr)
        # Fall back to plugin-provided commands (e.g. the GUI).
        ep = _plugin_entry_points().get(name)
        if ep is not None:
            try:
                return ep.load()
            except Exception as exc:
                raise click.ClickException(
                    f"Could not load the '{name}' command from plugin '{ep.value}': {exc}"
                )
        return None

    def shell_complete(self, ctx, incomplete):
        """Complete subcommand names from the static table without importing the
        backing modules (click's default would import every command for its help
        text, which is slow and pulls in graph-tool/pymeshlab/pycurv)."""
        from click.shell_completion import CompletionItem

        results = []
        for name in self.list_commands(ctx):
            if not name.startswith(incomplete):
                continue
            if name in _LAZY_COMMANDS:
                help_text = _LAZY_COMMANDS[name][1]
            else:
                cmd = click.Group.get_command(self, ctx, name)
                help_text = cmd.get_short_help_str() if cmd is not None else ""
            results.append(CompletionItem(name, help=help_text))
        # Add this group's own option completions (e.g. --help/--version) without
        # MultiCommand's command-importing behavior.
        results.extend(click.Command.shell_complete(self, ctx, incomplete))
        return results

    def _short_help(self, ctx, name):
        if name in _LAZY_COMMANDS:
            return _LAZY_COMMANDS[name][1]
        # Eager commands and plugin-provided commands: ask for the command object.
        cmd = self.get_command(ctx, name)
        return cmd.get_short_help_str() if cmd is not None else ""

    def format_commands(self, ctx, formatter):
        # List commands grouped into the standard pipeline order rather than
        # alphabetically, so the recommended sequence is clear at a glance.
        available = set(self.list_commands(ctx))
        listed = set()
        for title, names in _SECTIONS:
            rows = [(n, self._short_help(ctx, n)) for n in names if n in available]
            listed.update(n for n in names if n in available)
            if rows:
                with formatter.section(title):
                    formatter.write_dl(rows)
        extras = sorted(available - listed)
        if extras:
            with formatter.section("Other commands"):
                formatter.write_dl([(n, self._short_help(ctx, n)) for n in extras])


@click.group(cls=LazyGroup)
@click.version_option(package_name="surface-morphometrics", prog_name="morphometrics")
def cli():
    """Surface Morphometrics: quantify membrane surfaces from cryo-ET.

    Run a pipeline step as a subcommand, passing it a config.yml. Generate a
    starter config with `morphometrics new_config`.
    """


@click.command(name="new_config")
@click.option("-o", "--output", "output_path", default="config.yml",
              type=click.Path(), show_default=True,
              help="Where to write the config file.")
@click.option("--simple/--verbose", "simple", default=False,
              help="Write a minimal config with only the commonly-adjusted settings "
                   "(--simple), or the fully annotated template exposing every option "
                   "(--verbose, the default).")
@click.option("--force", is_flag=True, default=False,
              help="Overwrite the output file if it already exists.")
def new_config(output_path, simple, force):
    """Write a template config.yml into the current directory.

    Omitted settings fall back to documented defaults, so a --simple config is fully
    runnable; use --verbose to see and tune every available option.
    """
    from importlib.resources import files

    if os.path.exists(output_path) and not force:
        raise click.ClickException(
            f"{output_path} already exists; use --force to overwrite."
        )
    template_name = "config_template_simple.yml" if simple else "config_template.yml"
    template = files("surface_morphometrics").joinpath(template_name).read_text()
    with open(output_path, "w") as handle:
        handle.write(template)
    kind = "minimal" if simple else "full"
    click.echo(f"Wrote {kind} config to {output_path}. Edit it to point at your data "
               "and set parameters.")
    if simple:
        click.echo("Omitted settings use sensible defaults; run "
                   "`morphometrics new_config --verbose` to see every option.")


cli.add_command(new_config)


@cli.result_callback()
def _print_next_step(result, **kwargs):
    """After a subcommand finishes successfully, print a hint for the next step."""
    ctx = click.get_current_context()
    hint = _NEXT_HINTS.get(ctx.invoked_subcommand)
    if hint:
        click.echo("\n" + "─" * 70)
        click.echo(hint)


# Entry point referenced by pyproject [project.scripts].
main = cli


if __name__ == "__main__":
    main()
