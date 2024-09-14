import argparse
from utils import V2BConverter

parser = argparse.ArgumentParser()

parser.add_argument("input_file", help="Path to the video file to convert.")
parser.add_argument("--output_file", help="Name of the output bag file.")
parser.add_argument("--output_dir", help="Directory of the output bag file.")
parser.add_argument("--sleep_rate", help="Time interval between video frames.")
parser.add_argument("--div_num", help="Skip cycle of video frames.")
parser.add_argument(
    "--calibration_file",
    help="Path to the calib-camchain.yaml outputted by kalibr.")
parser.add_argument(
    "--do_not_save_images",
    action='store_false',
    dest="save_images",
    help=("If passed, the extracted frames will not be written to file as "
          "images, but only saved in the output rosbag."))
parser.add_argument("--verbose",
                    action='store_true',
                    help="Whether to have verbose prints.")
args = parser.parse_args()

converter = V2BConverter(
    args.input_file, args.output_file, **{
        'output_dir': args.output_dir,
        'sleep_rate': args.sleep_rate,
        'div_num': args.div_num,
        'calibration_file': args.calibration_file,
        'save_images': args.save_images,
        'verbose': args.verbose
    })
converter.convert()
