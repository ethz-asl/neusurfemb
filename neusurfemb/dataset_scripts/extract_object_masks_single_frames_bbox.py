import argparse
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import torch  # NOQA
import onnxruntime
from scipy.special import expit
import skimage

from segment_anything import sam_model_registry, SamPredictor

selected_points = []
first_image = None


def multi_hyp_sam(predictor,
                  image_embedding,
                  bbox,
                  orig_im_size,
                  mask_threshold=None,
                  vis=False):
    num_hypotheses = 1
    # Compute `num_hypotheses` hypotheses. For each hypothesis, take the
    # different in-mask points found so far and use as prompt the mean point of
    # them, perturbed based on their standard deviation.

    if (mask_threshold is None):
        mask_threshold = predictor.model.mask_threshold

    all_regressed_logits = []
    all_scores = []
    # For each hypothesis, just use a single point prompt.

    # transformed_bbox = predictor.transform.apply_boxes(
    #     bbox, tuple(orig_im_size)).astype(np.float32)
    onnx_coord = bbox.reshape(2, 2)[None, :, :]
    onnx_label = np.array([2, 3])[None, :].astype(np.float32)
    onnx_coord = predictor.transform.apply_coords(
        onnx_coord, tuple(orig_im_size)).astype(np.float32)

    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)
    ort_inputs = {
        "image_embeddings": image_embedding,
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": onnx_mask_input,
        "has_mask_input": onnx_has_mask_input,
        "orig_im_size": orig_im_size
    }

    regressed_logits, scores, _ = ort_session.run(None, ort_inputs)
    regressed_logits = regressed_logits[0]
    scores = scores[0]
    # Select the largest mask.
    num_pixels_per_mask = np.count_nonzero(
        (expit(regressed_logits)
         > mask_threshold).reshape(regressed_logits.shape[0], -1),
        axis=-1)

    regressed_logits = regressed_logits[np.argmax(num_pixels_per_mask)]
    scores = scores[np.argmax(num_pixels_per_mask)]

    all_regressed_logits.append(regressed_logits)
    all_scores.append(scores)

    all_regressed_logits = np.stack(all_regressed_logits)
    all_scores = np.stack(all_scores)

    num_pixels_per_mask = np.count_nonzero(
        (expit(all_regressed_logits)
         > mask_threshold).reshape(all_regressed_logits.shape[0], -1),
        axis=-1)
    # Keep the largest mask.
    regressed_mask = expit(
        all_regressed_logits[np.argmax(num_pixels_per_mask)]) > mask_threshold

    # Keep only the largest connect component in the mask.
    connected_components = skimage.measure.label(regressed_mask.copy(),
                                                 background=0)
    labels_conn_comp = np.unique(connected_components)
    assert (np.all(labels_conn_comp == np.arange(len(labels_conn_comp))))

    assert (labels_conn_comp[0] == 0)
    if (len(labels_conn_comp) > 2):
        largest_label = 1 + np.argmax([
            np.count_nonzero(connected_components == label)
            for label in labels_conn_comp[1:]
        ])
        regressed_mask = connected_components == largest_label
    else:
        regressed_mask = (connected_components == 1)

    if (vis):
        plt.imshow(regressed_mask)
        plt.show()

    return regressed_mask


def click_callback(event, x, y, flags, param):
    """Based on https://pyimagesearch.com/2015/03/09/capturing-mouse-click-
    events-with-python-and-opencv/. """
    global selected_points, first_image
    if (event == cv2.EVENT_LBUTTONDOWN):
        selected_points.append((x, y))
    elif (event == cv2.EVENT_LBUTTONUP):
        first_image = cv2.circle(first_image, selected_points[-1], 1,
                                 (0, 0, 255), 3)
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.imshow(window_title, first_image)


parser = argparse.ArgumentParser()

parser.add_argument('--image-path', required=True)
parser.add_argument('--vis', action='store_true', help="Debug visualization.")
parser.add_argument('--bbox-prompt',
                    nargs='+',
                    type=int,
                    help=("Bounding box prompt in the first image (x_min, "
                          "y_min, x_max, y_max)."))

args = parser.parse_args()

image_path = args.image_path

dataset_folder = os.path.dirname(os.path.dirname(image_path))

masked_image_folder = os.path.join(dataset_folder, "rgb")
masked_depth_folder = os.path.join(dataset_folder, "depth")

unmasked_image_folder = os.path.join(dataset_folder, "rgb_unmasked")
assert (os.path.exists(unmasked_image_folder))

if (os.path.exists(masked_depth_folder)):
    unmasked_depth_folder = os.path.join(dataset_folder, "depth_unmasked")
    has_depth = True
else:
    has_depth = False

# Read the image.
image = cv2.imread(image_path)
if (has_depth):
    depth_image = cv2.imread(
        os.path.join(unmasked_depth_folder, os.path.basename(image_path)),
        cv2.IMREAD_UNCHANGED)

else:
    depth_image = None

# Instantiate SAM model.
model_checkpoints = sorted(
    glob.glob(
        os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "third_party", "segment-anything", "weights", "*.pth")))
assert (len(model_checkpoints) == 1)
device = "cuda:0"

sam = sam_model_registry["vit_h"](checkpoint=model_checkpoints[0]).to(
    device=device).eval()
predictor = SamPredictor(sam)

# Load ONNX quantized model.
onnx_model_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "third_party", "segment-anything",
    "weights", "sam_onnx_quantized.onnx")
ort_session = onnxruntime.InferenceSession(onnx_model_path,
                                           providers=['CUDAExecutionProvider'])

points3D_in_mask = set()
points3D_outside_mask = set()
object_masks = []
all_masked_images = []

H, W = image.shape[:2]

if (args.bbox_prompt is None):
    # Ask the user to select a bounding box around the object in the image.
    window_title = ("Please select the bounding box, then press Enter")
    first_image = image.copy()
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    bbox = np.array(cv2.selectROI(window_title, first_image))
    bbox[2] += bbox[0]
    bbox[3] += bbox[1]
else:
    bbox = np.array(args.bbox_prompt)

# Run inference of SAM on the current image, based on the prompts.
predictor.set_image(image)

# - Get image embedding.
image_embedding = predictor.get_image_embedding().cpu().numpy()

# - Regress a mask, using potentially multiple prompt hypotheses.
regressed_mask = multi_hyp_sam(predictor=predictor,
                               image_embedding=image_embedding,
                               bbox=bbox,
                               orig_im_size=np.array(image.shape[:2],
                                                     dtype=np.float32),
                               mask_threshold=0.5,
                               vis=args.vis)

if (args.vis):
    img_vis = np.concatenate(
        [image[..., ::-1],
         np.zeros_like(image[..., 0])[..., None]], axis=-1)
    img_vis[..., 3] = 255 * regressed_mask
    plt.imshow(img_vis)
    plt.show()

masked_image = np.concatenate(
    [image[..., ::-1], (255 * regressed_mask[..., None]).astype(np.uint8)],
    axis=-1)

# Save the masked image.
cv2.imwrite(filename=os.path.join(masked_image_folder,
                                  os.path.basename(image_path)),
            img=masked_image[..., [2, 1, 0, 3]])

if (has_depth):
    # Save the masked depth image.
    masked_depth_image = depth_image.copy()
    masked_depth_image[np.logical_not(regressed_mask)] = 0.
    cv2.imwrite(filename=os.path.join(masked_depth_folder,
                                      os.path.basename(image_path)),
                img=masked_depth_image)
