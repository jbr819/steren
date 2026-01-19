# src/gwel/cli.py
import os
import json
import yaml
import typer
from steren.data import HRimage



app = typer.Typer(add_completion = True, help="STEREN CLI tool", invoke_without_command=True, context_settings={"help_option_names": ["-h", "--help"]})

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError  # For Python <3.8

try:
    VERSION = version("steren")
except PackageNotFoundError:
    VERSION = "unknown"


ASCII_ART = r"""
   _____________________  _______   __
  / ___/_  __/ ____/ __ \/ ____/ | / /
  \__ \ / / / __/ / /_/ / __/ /  |/ / 
 ___/ // / / /___/ _, _/ /___/ /|  /  
/____//_/ /_____/_/ |_/_____/_/ |_/   
                                      """


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version_flag: bool = typer.Option(
        False, "--version", "-v", help="Show STEREN CLI version"
    )):
    """
    Callback function that runs when 'steren' is called without a subcommand.
    """

    if version_flag:
        typer.secho(f"STEREN CLI version {VERSION}", fg=typer.colors.GREEN, bold=True)
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        print(ASCII_ART)
        typer.secho(f"STEREN CLI version {VERSION}", fg=typer.colors.CYAN, bold=True)
        typer.echo("Available commands:")

        # Use rich-like table formatting with Typer
        for name, cmd in ctx.command.commands.items():
            description = cmd.help or cmd.callback.__doc__ or ""
            if description:
                description = description.strip().split("\n")[0]
            typer.secho(f"  {name:<10}", fg=typer.colors.YELLOW, bold=True, nl=False)
            typer.echo(f" {description}")

@app.command()
def search(
    ctx: typer.Context, 
        type: str= typer.Argument(help="Type of planetary data [Supported types: HiRISE].")):
    """
    Search planetary data catalogues.
    """
    match type:
        case 'HiRISE':
            HRimage().search()


@app.command()
def metadata(
    ctx: typer.Context,
    type: str= typer.Argument(help="Type of planetary data [Supported types: HiRISE]."),
    identifier: str= typer.Argument(help ="Identifier of the planetary data")
    ):
    """
    Obtain planetary data metadata.
    """
    match type:
        case "HiRISE":
            hrimage = HRimage(identifier)
            metadata = hrimage.meta()
            print(yaml.dump(metadata, default_flow_style=False))
        case _:
            typer.secho(f"Data type {type} is not supported.")
         


@app.command()
def download(
    ctx: typer.Context,
    type: str= typer.Argument(help="Type of planetary data [Supported types: HiRISE]."),
    identifier: str= typer.Argument(help ="Identifier of the planetary data"),
    output_dir: str= typer.Option(".","-o","--output_dir",help="Directory path to download data into.")
        ):
    """
    Download planetary data.
    """
    match type:
        case "HiRISE":
            hrimage = HRimage(identifier)
            hrimage.download(".")
        case _:
            typer.secho(f"Data type {type} is not supported.")
       

@app.command()
def detect(
    ctx: typer.Context,
    data_type: str= typer.Argument(help="Type of planetary data [Supported types: HiRISE]."),
    detector_type: str= typer.Argument(help="Type of detector [Supported types: YOLOv8]."),
    model_weights: str=typer.Argument(help="Path to model weights."),
    img_path: str=typer.Argument(help="Path to image data."),
    output_file: str=typer.Argument(help="Output file path [Supported types: Geopackage (.gpkg), Shapefile (.shp), Tabular (.csv)]"),
    tile_size: int= typer.Option(512,'-ts','--tile_size', help="Tile size"),
    overlap: int= typer.Option(100,'-o','--overlap', help="Overlap size"),
    conf_thresh: float=typer.Option(0.25,'-c','--confidence',help="Minimum confidence threshold of detections."),
    iou_thresh: float=typer.Option(0.45, '-iou','--intersection_over_union', help="Threshold for the maximum intersection over union between detections.")):
    """
    Detect instances within planetary data with a pre-trained model.
    """
    match (data_type, detector_type):
        case ('HiRISE','YOLOv8'):

            from steren.networks.YOLOv8 import YOLOv8
            detector = YOLOv8(model_weights,
                                tile_size, 
                                overlap,
                                conf_thresh,
                                iou_thresh)

            hrimage = HRimage(img_path = img_path)
            hrimage.detect(detector) 
            hrimage.export_detections(output_file)





@app.command()
def slice(
    ctx: typer.Context,
    type: str= typer.Argument(help="Type of planetary data [Supported types: HiRISE]."),
    detector_type: str= typer.Argument(help="Type of detector [Supported types: YOLOv8]."),
    ):
    """
    Slice planetary data to become a regular image data set.
    """
    pass

if __name__ == "__main__":
    app()

