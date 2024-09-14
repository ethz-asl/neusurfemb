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
                  prompt_pixels_in_mask,
                  prompt_pixels_outside_mask,
                  orig_im_size,
                  mask_threshold=None,
                  num_hypotheses=5,
                  vis=False):
    num_hypotheses = min(num_hypotheses, len(prompt_pixels_in_mask))

    num_points_for_average = len(prompt_pixels_in_mask)
    # Compute `num_hypotheses` hypotheses. For each hypothesis, take the
    # different in-mask points found so far and use as prompt the mean point of
    # them, perturbed based on their standard deviation.

    prompt_hypotheses = []
    prompt_pixels_in_mask_arr = np.array(prompt_pixels_in_mask)
    for _ in range(num_hypotheses):
        random_perm = np.random.permutation(
            len(prompt_pixels_in_mask))[:num_points_for_average]
        curr_prompt_hypothesis = (
            prompt_pixels_in_mask_arr[random_perm].mean(axis=0) +
            ((np.random.rand(2) * 2.) - 1.) *
            prompt_pixels_in_mask_arr[random_perm].std(axis=0)).astype(int)
        prompt_hypotheses.append(curr_prompt_hypothesis)

    if (mask_threshold is None):
        mask_threshold = predictor.model.mask_threshold

    all_regressed_logits = []
    all_scores = []
    for prompt_idx in range(num_hypotheses):
        # For each hypothesis, just use a single point prompt.
        curr_prompt_pixels_in_mask = ([prompt_hypotheses[prompt_idx]] +
                                      prompt_pixels_outside_mask[:1])
        input_points = np.array(curr_prompt_pixels_in_mask)
        input_labels = np.array([1.] +
                                [0. for _ in prompt_pixels_outside_mask[:1]])

        onnx_coord = np.concatenate(
            [input_points, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
        onnx_label = np.concatenate([input_labels, np.array([-1])],
                                    axis=0)[None, :].astype(np.float32)

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
    max_num_columns = 5
    num_rows = max(1, num_hypotheses // max_num_columns)
    num_columns = (num_hypotheses
                   if num_hypotheses <= max_num_columns else max_num_columns)
    if (vis):
        _, ax = plt.subplots(num_rows, num_columns)
        for prompt_idx in range(num_hypotheses):
            curr_img = expit(all_regressed_logits[prompt_idx]) > mask_threshold
            curr_img = (255 * np.repeat(curr_img[..., None], 3, -1)).astype(
                np.uint8)
            curr_img = cv2.circle(img=curr_img.copy(),
                                  center=prompt_hypotheses[prompt_idx],
                                  radius=5,
                                  color=[255, 0., 0., 255.])

            if (num_rows > 1):
                ax[prompt_idx // num_columns,
                   prompt_idx % num_columns].imshow(curr_img)
            else:
                if (num_rows == num_columns == 1):
                    ax.imshow(curr_img)
                else:
                    ax[prompt_idx % num_columns].imshow(curr_img)

        plt.show()

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

parser.add_argument('--image-path', required=True)
parser.add_argument('--initial-prompt',
                    nargs='+',
                    type=int,
                    help="Positive prompt in the image (x, y).")
parser.add_argument('--vis', action='store_true', help="Debug visualization.")

args = parser.parse_args()

if (args.initial_prompt is None):
    initial_prompt = None
else:
    initial_prompt = tuple(args.initial_prompt)

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

# Ask the user to select one or more points belonging to the object in the
# image.
if (initial_prompt is None):
    window_title = ("Please select one or more in-object points, then press "
                    "Q")
    first_image = image.copy()
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_title, click_callback)
    while True:
        cv2.imshow(window_title, first_image)
        key = cv2.waitKey(1) & 0xFF
        if (key == ord("q")):
            break
    cv2.destroyAllWindows()
else:
    selected_points = [initial_prompt]
prompt_pixels_in_mask = selected_points
prompt_pixels_outside_mask = []

# Run inference of SAM on the current image, based on the prompts.
predictor.set_image(image)

# - Get image embedding.
image_embedding = predictor.get_image_embedding().cpu().numpy()

# - Regress a mask, using potentially multiple prompt hypotheses.
regressed_mask = multi_hyp_sam(
    predictor=predictor,
    image_embedding=image_embedding,
    prompt_pixels_in_mask=prompt_pixels_in_mask,
    prompt_pixels_outside_mask=prompt_pixels_outside_mask,
    orig_im_size=np.array(image.shape[:2], dtype=np.float32),
    num_hypotheses=len(prompt_pixels_in_mask),
    mask_threshold=0.5,
    vis=args.vis)

if (args.vis):
    img_vis = np.concatenate(
        [image, np.zeros_like(image[..., 0])[..., None]], axis=-1)
    img_vis[..., 3] = 255 * regressed_mask
    prompt_pixels_in_mask = np.array(prompt_pixels_in_mask)
    img_vis[prompt_pixels_in_mask[..., 1],
            prompt_pixels_in_mask[..., 0]] = [0, 255, 0, 255]
    if (len(prompt_pixels_outside_mask) > 0):
        prompt_pixels_outside_mask = np.array(prompt_pixels_outside_mask)
        img_vis[prompt_pixels_outside_mask[..., 1],
                prompt_pixels_outside_mask[..., 0]] = [255, 0, 0, 255]
    plt.imshow(img_vis)
    plt.show()

masked_image = np.concatenate(
    [image[..., ::-1], (255 * regressed_mask[..., None]).astype(np.uint8)],
    axis=-1)

# Save the masked image.
cv2.imwrite(filename=os.path.join(masked_image_folder,
                                  os.path.basename(image_path)),
            img=masked_image[..., [2, 1, 0, 3]])

# Save the masked depth image.
if (has_depth):
    masked_depth_image = depth_image.copy()
    masked_depth_image[np.logical_not(regressed_mask)] = 0.
    cv2.imwrite(filename=os.path.join(masked_depth_folder,
                                      os.path.basename(image_path)),
                img=masked_depth_image)
