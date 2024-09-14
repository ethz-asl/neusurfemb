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
import tqdm

import mmtrack
from mmtrack.apis import inference_sot, init_model
from segment_anything import sam_model_registry, SamPredictor

selected_points = []
first_image = None


def skew(x):
    x = x.flatten()
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def multi_hyp_sam(predictor,
                  image_embedding,
                  bbox,
                  orig_im_size,
                  mask_threshold=None,
                  vis=False):
    if (mask_threshold is None):
        mask_threshold = predictor.model.mask_threshold

    all_regressed_logits = []
    all_scores = []
    # For each hypothesis, just use a single point prompt.

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
        cv2.imshow(window_title, first_image)


parser = argparse.ArgumentParser()

parser.add_argument('--dataset-folder', required=True)
parser.add_argument('--vis', action='store_true', help="Debug visualization.")
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--bbox-prompt',
                    nargs='+',
                    type=int,
                    help=("Bounding box prompt in the first image (x_min, "
                          "y_min, x_max, y_max)."))
parser.add_argument('--bbox-scale-factor',
                    type=float,
                    default=1.1,
                    help=("If not None, the bounding box tracked from the "
                          "previous frame is enlarged by this factor."))

args = parser.parse_args()

dataset_folder = args.dataset_folder
bbox_scale_factor = args.bbox_scale_factor

unmasked_image_folder = os.path.join(dataset_folder, "rgb")
unmasked_depth_folder = os.path.join(dataset_folder, "depth")

unmasked_image_folder_new = os.path.join(dataset_folder, "rgb_unmasked")
os.rename(unmasked_image_folder, unmasked_image_folder_new)
masked_image_folder = unmasked_image_folder
os.makedirs(masked_image_folder)
if (os.path.exists(unmasked_depth_folder)):
    unmasked_depth_folder_new = os.path.join(dataset_folder, "depth_unmasked")
    os.rename(unmasked_depth_folder, unmasked_depth_folder_new)
    masked_depth_folder = unmasked_depth_folder
    os.makedirs(masked_depth_folder)
    has_depth = True
else:
    has_depth = False

image_paths = sorted(glob.glob(os.path.join(unmasked_image_folder_new, "*")))

images = [
    cv2.imread(image_path, cv2.IMREAD_UNCHANGED) for image_path in image_paths
]
if (has_depth):
    depth_image_paths = sorted(
        glob.glob(os.path.join(unmasked_depth_folder_new, "*")))
    assert (len(depth_image_paths) == len(image_paths))
    assert (np.all([
        os.path.basename(depth_path) == os.path.basename(image_path)
        for depth_path, image_path in zip(depth_image_paths, image_paths)
    ]))
    depth_images = [
        cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        for image_path in depth_image_paths
    ]
else:
    depth_images = [None for _ in images]

# Instantiate object tracker.
sot_config = os.path.join(os.path.dirname(os.path.dirname(mmtrack.__file__)),
                          'configs/sot/mixformer/mixformer_cvt_500e_got10k.py')
sot_checkpoint = os.path.join(
    os.path.dirname(os.path.dirname(mmtrack.__file__)),
    'checkpoints/mixformer_cvt_500e_got10k.pth')
sot_model = init_model(sot_config, sot_checkpoint, device='cuda:0')

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

is_first_image = True

H = None
W = None

# Track the bounding box, updating it each time with the tight bounding box to
# the regressed object mask.
for image_idx in tqdm.tqdm(range(len(images))):
    image = images[image_idx]
    if (has_depth):
        depth_image = depth_images[image_idx]

    if (H is None):
        H, W = image.shape[:2]
    else:
        assert (image.shape[:2] == (H, W))

    if (is_first_image):
        if (args.bbox_prompt is None):
            # Ask the user to select a bounding box around the object in the first
            # image.
            window_title = ("Please select the bounding box, then press Enter")
            first_image = images[0].copy()
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
            bbox = np.array(cv2.selectROI(window_title, first_image))
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
        else:
            bbox = np.array(args.bbox_prompt)
    else:
        # Get the bounding box tracked from the previous frame(s).
        bbox = inference_sot(sot_model, image, bbox, frame_id=image_idx -
                             1)['track_bboxes'][:4].astype(int)
        if (bbox_scale_factor is not None):
            # - Enlarge the bounding box if requested.
            x_min, y_min, x_max, y_max = bbox
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2.

            x_min = max(0, x_center + (x_min - x_center) * bbox_scale_factor)
            x_max = min(x_center + (x_max - x_center) * bbox_scale_factor,
                        W - 1)
            y_min = max(0, y_center + (y_min - y_center) * bbox_scale_factor)
            y_max = min(y_center + (y_max - y_center) * bbox_scale_factor,
                        H - 1)
            bbox = np.array([x_min, y_min, x_max, y_max])

    # - Run inference of SAM on the current image, based on the bounding
    #   "tracked" from the previous image.
    predictor.set_image(image)

    # - Get image embedding.
    image_embedding = predictor.get_image_embedding().cpu().numpy()

    # - Regress a mask, using the bounding box.
    regressed_mask = multi_hyp_sam(predictor=predictor,
                                   image_embedding=image_embedding,
                                   bbox=bbox,
                                   orig_im_size=np.array(image.shape[:2],
                                                         dtype=np.float32),
                                   mask_threshold=0.5,
                                   vis=args.vis)

    # - Use the current mask to define the bounding box to be tracked in the
    #   subsequent image.
    bbox_tracked = bbox.copy()
    y_where, x_where = np.where(regressed_mask)
    bbox = np.array(
        [x_where.min(),
         y_where.min(),
         x_where.max(),
         y_where.max()])

    if (args.vis):
        img_vis = np.concatenate(
            [image[..., ::-1],
             np.zeros_like(image[..., 0])[..., None]],
            axis=-1)
        img_vis[..., 3] = 255 * regressed_mask
        plt.imshow(img_vis)
        plt.show()

    masked_image = np.concatenate(
        [image[..., ::-1], (255 * regressed_mask[..., None]).astype(np.uint8)],
        axis=-1)

    if (args.vis):
        (x_min_tracked, y_min_tracked, x_max_tracked,
         y_max_tracked) = bbox_tracked.astype(int)
        x_min, y_min, x_max, y_max = bbox
        # Final bbox.
        image_with_bboxes = cv2.line(masked_image.copy(),
                                     pt1=(x_min, y_min),
                                     pt2=(x_max, y_min),
                                     color=(255, 0, 0, 255),
                                     thickness=3)
        image_with_bboxes = cv2.line(image_with_bboxes,
                                     pt1=(x_max, y_min),
                                     pt2=(x_max, y_max),
                                     color=(255, 0, 0, 255),
                                     thickness=3)
        image_with_bboxes = cv2.line(image_with_bboxes,
                                     pt1=(x_max, y_max),
                                     pt2=(x_min, y_max),
                                     color=(255, 0, 0, 255),
                                     thickness=3)
        image_with_bboxes = cv2.line(image_with_bboxes,
                                     pt1=(x_min, y_max),
                                     pt2=(x_min, y_min),
                                     color=(255, 0, 0, 255),
                                     thickness=3)
        # Tracked bbox.
        image_with_bboxes = cv2.line(image_with_bboxes,
                                     pt1=(x_min_tracked, y_min_tracked),
                                     pt2=(x_max_tracked, y_min_tracked),
                                     color=(0, 0, 255, 255),
                                     thickness=3)
        image_with_bboxes = cv2.line(image_with_bboxes,
                                     pt1=(x_max_tracked, y_min_tracked),
                                     pt2=(x_max_tracked, y_max_tracked),
                                     color=(0, 0, 255, 255),
                                     thickness=3)
        image_with_bboxes = cv2.line(image_with_bboxes,
                                     pt1=(x_max_tracked, y_max_tracked),
                                     pt2=(x_min_tracked, y_max_tracked),
                                     color=(0, 0, 255, 255),
                                     thickness=3)
        image_with_bboxes = cv2.line(image_with_bboxes,
                                     pt1=(x_min_tracked, y_max_tracked),
                                     pt2=(x_min_tracked, y_min_tracked),
                                     color=(0, 0, 255, 255),
                                     thickness=3)
        plt.imshow(image_with_bboxes)
        plt.show()

    # - Save the masked image.
    image_path = image_paths[image_idx]
    masked_image_path = os.path.join(masked_image_folder,
                                     os.path.basename(image_path))
    cv2.imwrite(filename=masked_image_path, img=masked_image[..., [2, 1, 0, 3]])

    if (args.verbose):
        print(f"Saved image '{masked_image_path}'.")

    if (has_depth):
        # Save the masked depth image.
        masked_depth_image_path = os.path.join(masked_depth_folder,
                                               os.path.basename(image_path))
        masked_depth_image = depth_image.copy()
        masked_depth_image[np.logical_not(regressed_mask)] = 0.
        cv2.imwrite(filename=masked_depth_image_path, img=masked_depth_image)
        if (args.verbose):
            print(f"Saved image '{masked_depth_image_path}'.")

    if (is_first_image):
        is_first_image = False
