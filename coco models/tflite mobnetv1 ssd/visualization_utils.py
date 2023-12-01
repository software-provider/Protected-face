
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
# Set headless-friendly backend.
import matplotlib; matplotlib.use('Agg')  # pylint: disable=multiple-statements
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import six
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf

from object_detection.core import keypoint_ops
from object_detection.core import standard_fields as fields
from object_detection.utils import shape_utils

_TITLE_LEFT_MARGIN = 10
_TITLE_TOP_MARGIN = 10
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def _get_multiplier_for_color_randomness():
 
  num_colors = len(STANDARD_COLORS)
  prime_candidates = [5, 7, 11, 13, 17]

  # Remove all prime candidates that divide the number of colors.
  prime_candidates = [p for p in prime_candidates if num_colors % p]
  if not prime_candidates:
    return 1

  # Return the closest prime number to num_colors / 10.
  abs_distance = [np.abs(num_colors / 10. - p) for p in prime_candidates]
  num_candidates = len(abs_distance)
  inds = [i for _, i in sorted(zip(abs_distance, range(num_candidates)))]
  return prime_candidates[inds[0]]


def save_image_array_as_png(image, output_path):
 
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  with tf.gfile.Open(output_path, 'w') as fid:
    image_pil.save(fid, 'PNG')


def encode_image_array_as_png_str(image):
 
  image_pil = Image.fromarray(np.uint8(image))
  output = six.BytesIO()
  image_pil.save(output, format='PNG')
  png_string = output.getvalue()
  output.close()
  return png_string


def draw_bounding_box_on_image_array(image,
                                     ymin,
                                     xmin,
                                     ymax,
                                     xmax,
                                     color='red',
                                     thickness=4,
                                     display_str_list=(),
                                     use_normalized_coordinates=True):

  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                             thickness, display_str_list,
                             use_normalized_coordinates)
  np.copyto(image, np.array(image_pil))


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
 
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  if use_normalized_coordinates:
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
  else:
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
  if thickness > 0:
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
               (left, top)],
              width=thickness,
              fill=color)
  try:
    font = ImageFont.truetype('arial.ttf', 24)
  except IOError:
    font = ImageFont.load_default()

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = bottom + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                          text_bottom)],
        fill=color)
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill='black',
        font=font)
    text_bottom -= text_height - 2 * margin


def draw_bounding_boxes_on_image_array(image,
                                       boxes,
                                       color='red',
                                       thickness=4,
                                       display_str_list_list=()):

  image_pil = Image.fromarray(image)
  draw_bounding_boxes_on_image(image_pil, boxes, color, thickness,
                               display_str_list_list)
  np.copyto(image, np.array(image_pil))


def draw_bounding_boxes_on_image(image,
                                 boxes,
                                 color='red',
                                 thickness=4,
                                 display_str_list_list=()):
 
  boxes_shape = boxes.shape
  if not boxes_shape:
    return
  if len(boxes_shape) != 2 or boxes_shape[1] != 4:
    raise ValueError('Input must be of size [N, 4]')
  for i in range(boxes_shape[0]):
    display_str_list = ()
    if display_str_list_list:
      display_str_list = display_str_list_list[i]
    draw_bounding_box_on_image(image, boxes[i, 0], boxes[i, 1], boxes[i, 2],
                               boxes[i, 3], color, thickness, display_str_list)


def create_visualization_fn(category_index,
                            include_masks=False,
                            include_keypoints=False,
                            include_keypoint_scores=False,
                            include_track_ids=False,
                            **kwargs):
  
  def visualization_py_func_fn(*args):
   
    image = args[0]
    boxes = args[1]
    classes = args[2]
    scores = args[3]
    masks = keypoints = keypoint_scores = track_ids = None
    pos_arg_ptr = 4  # Positional argument for first optional tensor (masks).
    if include_masks:
      masks = args[pos_arg_ptr]
      pos_arg_ptr += 1
    if include_keypoints:
      keypoints = args[pos_arg_ptr]
      pos_arg_ptr += 1
    if include_keypoint_scores:
      keypoint_scores = args[pos_arg_ptr]
      pos_arg_ptr += 1
    if include_track_ids:
      track_ids = args[pos_arg_ptr]

    return visualize_boxes_and_labels_on_image_array(
        image,
        boxes,
        classes,
        scores,
        category_index=category_index,
        instance_masks=masks,
        keypoints=keypoints,
        keypoint_scores=keypoint_scores,
        track_ids=track_ids,
        **kwargs)
  return visualization_py_func_fn


def draw_heatmaps_on_image(image, heatmaps):
  
  draw = ImageDraw.Draw(image)
  channel = heatmaps.shape[2]
  for c in range(channel):
    heatmap = heatmaps[:, :, c] * 255
    heatmap = heatmap.astype('uint8')
    bitmap = Image.fromarray(heatmap, 'L')
    bitmap.convert('1')
    draw.bitmap(
        xy=[(0, 0)],
        bitmap=bitmap,
        fill=STANDARD_COLORS[c])


def draw_heatmaps_on_image_array(image, heatmaps):
 
  if not isinstance(image, np.ndarray):
    image = image.numpy()
  if not isinstance(heatmaps, np.ndarray):
    heatmaps = heatmaps.numpy()
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  draw_heatmaps_on_image(image_pil, heatmaps)
  return np.array(image_pil)


def draw_heatmaps_on_image_tensors(images,
                                   heatmaps,
                                   apply_sigmoid=False):
 
  # Additional channels are being ignored.
  if images.shape[3] > 3:
    images = images[:, :, :, 0:3]
  elif images.shape[3] == 1:
    images = tf.image.grayscale_to_rgb(images)

  _, height, width, _ = shape_utils.combined_static_and_dynamic_shape(images)
  if apply_sigmoid:
    heatmaps = tf.math.sigmoid(heatmaps)
  resized_heatmaps = tf.image.resize(heatmaps, size=[height, width])

  elems = [images, resized_heatmaps]

  def draw_heatmaps(image_and_heatmaps):
    """Draws heatmaps on image."""
    image_with_heatmaps = tf.py_function(
        draw_heatmaps_on_image_array,
        image_and_heatmaps,
        tf.uint8)
    return image_with_heatmaps
  images = tf.map_fn(draw_heatmaps, elems, dtype=tf.uint8, back_prop=False)
  return images


def _resize_original_image(image, image_shape):
  image = tf.expand_dims(image, 0)
  image = tf.image.resize_images(
      image,
      image_shape,
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
      align_corners=True)
  return tf.cast(tf.squeeze(image, 0), tf.uint8)


def draw_bounding_boxes_on_image_tensors(images,
                                         boxes,
                                         classes,
                                         scores,
                                         category_index,
                                         original_image_spatial_shape=None,
                                         true_image_shape=None,
                                         instance_masks=None,
                                         keypoints=None,
                                         keypoint_scores=None,
                                         keypoint_edges=None,
                                         track_ids=None,
                                         max_boxes_to_draw=20,
                                         min_score_thresh=0.2,
                                         use_normalized_coordinates=True):
 
  # Additional channels are being ignored.
  if images.shape[3] > 3:
    images = images[:, :, :, 0:3]
  elif images.shape[3] == 1:
    images = tf.image.grayscale_to_rgb(images)
  visualization_keyword_args = {
      'use_normalized_coordinates': use_normalized_coordinates,
      'max_boxes_to_draw': max_boxes_to_draw,
      'min_score_thresh': min_score_thresh,
      'agnostic_mode': False,
      'line_thickness': 4,
      'keypoint_edges': keypoint_edges
  }
  if true_image_shape is None:
    true_shapes = tf.constant(-1, shape=[images.shape.as_list()[0], 3])
  else:
    true_shapes = true_image_shape
  if original_image_spatial_shape is None:
    original_shapes = tf.constant(-1, shape=[images.shape.as_list()[0], 2])
  else:
    original_shapes = original_image_spatial_shape

  visualize_boxes_fn = create_visualization_fn(
      category_index,
      include_masks=instance_masks is not None,
      include_keypoints=keypoints is not None,
      include_keypoint_scores=keypoint_scores is not None,
      include_track_ids=track_ids is not None,
      **visualization_keyword_args)

  elems = [true_shapes, original_shapes, images, boxes, classes, scores]
  if instance_masks is not None:
    elems.append(instance_masks)
  if keypoints is not None:
    elems.append(keypoints)
  if keypoint_scores is not None:
    elems.append(keypoint_scores)
  if track_ids is not None:
    elems.append(track_ids)

  def draw_boxes(image_and_detections):
    """Draws boxes on image."""
    true_shape = image_and_detections[0]
    original_shape = image_and_detections[1]
    if true_image_shape is not None:
      image = shape_utils.pad_or_clip_nd(image_and_detections[2],
                                         [true_shape[0], true_shape[1], 3])
    if original_image_spatial_shape is not None:
      image_and_detections[2] = _resize_original_image(image, original_shape)

    image_with_boxes = tf.py_func(visualize_boxes_fn, image_and_detections[2:],
                                  tf.uint8)
    return image_with_boxes

  images = tf.map_fn(draw_boxes, elems, dtype=tf.uint8, back_prop=False)
  return images


def draw_side_by_side_evaluation_image(eval_dict,
                                       category_index,
                                       max_boxes_to_draw=20,
                                       min_score_thresh=0.2,
                                       use_normalized_coordinates=True,
                                       keypoint_edges=None):
 
  detection_fields = fields.DetectionResultFields()
  input_data_fields = fields.InputDataFields()

  images_with_detections_list = []

  # Add the batch dimension if the eval_dict is for single example.
  if len(eval_dict[detection_fields.detection_classes].shape) == 1:
    for key in eval_dict:
      if (key != input_data_fields.original_image and
          key != input_data_fields.image_additional_channels):
        eval_dict[key] = tf.expand_dims(eval_dict[key], 0)

  for indx in range(eval_dict[input_data_fields.original_image].shape[0]):
    instance_masks = None
    if detection_fields.detection_masks in eval_dict:
      instance_masks = tf.cast(
          tf.expand_dims(
              eval_dict[detection_fields.detection_masks][indx], axis=0),
          tf.uint8)
    keypoints = None
    keypoint_scores = None
    if detection_fields.detection_keypoints in eval_dict:
      keypoints = tf.expand_dims(
          eval_dict[detection_fields.detection_keypoints][indx], axis=0)
      if detection_fields.detection_keypoint_scores in eval_dict:
        keypoint_scores = tf.expand_dims(
            eval_dict[detection_fields.detection_keypoint_scores][indx], axis=0)
      else:
        keypoint_scores = tf.cast(keypoint_ops.set_keypoint_visibilities(
            keypoints), dtype=tf.float32)

    groundtruth_instance_masks = None
    if input_data_fields.groundtruth_instance_masks in eval_dict:
      groundtruth_instance_masks = tf.cast(
          tf.expand_dims(
              eval_dict[input_data_fields.groundtruth_instance_masks][indx],
              axis=0), tf.uint8)
    groundtruth_keypoints = None
    groundtruth_keypoint_scores = None
    gt_kpt_vis_fld = input_data_fields.groundtruth_keypoint_visibilities
    if input_data_fields.groundtruth_keypoints in eval_dict:
      groundtruth_keypoints = tf.expand_dims(
          eval_dict[input_data_fields.groundtruth_keypoints][indx], axis=0)
      if gt_kpt_vis_fld in eval_dict:
        groundtruth_keypoint_scores = tf.expand_dims(
            tf.cast(eval_dict[gt_kpt_vis_fld][indx], dtype=tf.float32), axis=0)
      else:
        groundtruth_keypoint_scores = tf.cast(
            keypoint_ops.set_keypoint_visibilities(
                groundtruth_keypoints), dtype=tf.float32)

    images_with_detections = draw_bounding_boxes_on_image_tensors(
        tf.expand_dims(
            eval_dict[input_data_fields.original_image][indx], axis=0),
        tf.expand_dims(
            eval_dict[detection_fields.detection_boxes][indx], axis=0),
        tf.expand_dims(
            eval_dict[detection_fields.detection_classes][indx], axis=0),
        tf.expand_dims(
            eval_dict[detection_fields.detection_scores][indx], axis=0),
        category_index,
        original_image_spatial_shape=tf.expand_dims(
            eval_dict[input_data_fields.original_image_spatial_shape][indx],
            axis=0),
        true_image_shape=tf.expand_dims(
            eval_dict[input_data_fields.true_image_shape][indx], axis=0),
        instance_masks=instance_masks,
        keypoints=keypoints,
        keypoint_scores=keypoint_scores,
        keypoint_edges=keypoint_edges,
        max_boxes_to_draw=max_boxes_to_draw,
        min_score_thresh=min_score_thresh,
        use_normalized_coordinates=use_normalized_coordinates)
    images_with_groundtruth = draw_bounding_boxes_on_image_tensors(
        tf.expand_dims(
            eval_dict[input_data_fields.original_image][indx], axis=0),
        tf.expand_dims(
            eval_dict[input_data_fields.groundtruth_boxes][indx], axis=0),
        tf.expand_dims(
            eval_dict[input_data_fields.groundtruth_classes][indx], axis=0),
        tf.expand_dims(
            tf.ones_like(
                eval_dict[input_data_fields.groundtruth_classes][indx],
                dtype=tf.float32),
            axis=0),
        category_index,
        original_image_spatial_shape=tf.expand_dims(
            eval_dict[input_data_fields.original_image_spatial_shape][indx],
            axis=0),
        true_image_shape=tf.expand_dims(
            eval_dict[input_data_fields.true_image_shape][indx], axis=0),
        instance_masks=groundtruth_instance_masks,
        keypoints=groundtruth_keypoints,
        keypoint_scores=groundtruth_keypoint_scores,
        keypoint_edges=keypoint_edges,
        max_boxes_to_draw=None,
        min_score_thresh=0.0,
        use_normalized_coordinates=use_normalized_coordinates)
    images_to_visualize = tf.concat([images_with_detections,
                                     images_with_groundtruth], axis=2)

    if input_data_fields.image_additional_channels in eval_dict:
      images_with_additional_channels_groundtruth = (
          draw_bounding_boxes_on_image_tensors(
              tf.expand_dims(
                  eval_dict[input_data_fields.image_additional_channels][indx],
                  axis=0),
              tf.expand_dims(
                  eval_dict[input_data_fields.groundtruth_boxes][indx], axis=0),
              tf.expand_dims(
                  eval_dict[input_data_fields.groundtruth_classes][indx],
                  axis=0),
              tf.expand_dims(
                  tf.ones_like(
                      eval_dict[input_data_fields.groundtruth_classes][indx],
                      dtype=tf.float32),
                  axis=0),
              category_index,
              original_image_spatial_shape=tf.expand_dims(
                  eval_dict[input_data_fields.original_image_spatial_shape]
                  [indx],
                  axis=0),
              true_image_shape=tf.expand_dims(
                  eval_dict[input_data_fields.true_image_shape][indx], axis=0),
              instance_masks=groundtruth_instance_masks,
              keypoints=None,
              keypoint_edges=None,
              max_boxes_to_draw=None,
              min_score_thresh=0.0,
              use_normalized_coordinates=use_normalized_coordinates))
      images_to_visualize = tf.concat(
          [images_to_visualize, images_with_additional_channels_groundtruth],
          axis=2)
    images_with_detections_list.append(images_to_visualize)

  return images_with_detections_list


def draw_densepose_visualizations(eval_dict,
                                  max_boxes_to_draw=20,
                                  min_score_thresh=0.2,
                                  num_parts=24,
                                  dp_coord_to_visualize=0):
 
  if dp_coord_to_visualize not in (0, 1):
    raise ValueError('`dp_coord_to_visualize` must be either 0 for v '
                     'coordinates), or 1 for u coordinates, but instead got '
                     '{}'.format(dp_coord_to_visualize))
  detection_fields = fields.DetectionResultFields()
  input_data_fields = fields.InputDataFields()

  if detection_fields.detection_masks not in eval_dict:
    raise ValueError('Expected `detection_masks` in `eval_dict`.')
  if detection_fields.detection_surface_coords not in eval_dict:
    raise ValueError('Expected `detection_surface_coords` in `eval_dict`.')

  images_with_detections_list = []
  for indx in range(eval_dict[input_data_fields.original_image].shape[0]):
    # Note that detection masks have already been resized to the original image
    # shapes, but `original_image` has not.
    # TODO(ronnyvotel): Consider resizing `original_image` in
    # eval_util.result_dict_for_batched_example().
    true_shape = eval_dict[input_data_fields.true_image_shape][indx]
    original_shape = eval_dict[
        input_data_fields.original_image_spatial_shape][indx]
    image = eval_dict[input_data_fields.original_image][indx]
    image = shape_utils.pad_or_clip_nd(image, [true_shape[0], true_shape[1], 3])
    image = _resize_original_image(image, original_shape)

    scores = eval_dict[detection_fields.detection_scores][indx]
    detection_masks = eval_dict[detection_fields.detection_masks][indx]
    surface_coords = eval_dict[detection_fields.detection_surface_coords][indx]

    def draw_densepose_py_func(image, detection_masks, surface_coords, scores):
      """Overlays part masks and surface coords on original images."""
      surface_coord_image = np.copy(image)
      for i, (score, surface_coord, mask) in enumerate(
          zip(scores, surface_coords, detection_masks)):
        if i == max_boxes_to_draw:
          break
        if score > min_score_thresh:
          draw_part_mask_on_image_array(image, mask, num_parts=num_parts)
          draw_float_channel_on_image_array(
              surface_coord_image, surface_coord[:, :, dp_coord_to_visualize],
              mask)
      return np.concatenate([image, surface_coord_image], axis=1)

    image_with_densepose = tf.py_func(
        draw_densepose_py_func,
        [image, detection_masks, surface_coords, scores],
        tf.uint8)
    images_with_detections_list.append(
        image_with_densepose[tf.newaxis, :, :, :])
  return images_with_detections_list


def draw_keypoints_on_image_array(image,
                                  keypoints,
                                  keypoint_scores=None,
                                  min_score_thresh=0.5,
                                  color='red',
                                  radius=2,
                                  use_normalized_coordinates=True,
                                  keypoint_edges=None,
                                  keypoint_edge_color='green',
                                  keypoint_edge_width=2):
 
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  draw_keypoints_on_image(image_pil,
                          keypoints,
                          keypoint_scores=keypoint_scores,
                          min_score_thresh=min_score_thresh,
                          color=color,
                          radius=radius,
                          use_normalized_coordinates=use_normalized_coordinates,
                          keypoint_edges=keypoint_edges,
                          keypoint_edge_color=keypoint_edge_color,
                          keypoint_edge_width=keypoint_edge_width)
  np.copyto(image, np.array(image_pil))


def draw_keypoints_on_image(image,
                            keypoints,
                            keypoint_scores=None,
                            min_score_thresh=0.5,
                            color='red',
                            radius=2,
                            use_normalized_coordinates=True,
                            keypoint_edges=None,
                            keypoint_edge_color='green',
                            keypoint_edge_width=2):
  
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  keypoints = np.array(keypoints)
  keypoints_x = [k[1] for k in keypoints]
  keypoints_y = [k[0] for k in keypoints]
  if use_normalized_coordinates:
    keypoints_x = tuple([im_width * x for x in keypoints_x])
    keypoints_y = tuple([im_height * y for y in keypoints_y])
  if keypoint_scores is not None:
    keypoint_scores = np.array(keypoint_scores)
    valid_kpt = np.greater(keypoint_scores, min_score_thresh)
  else:
    valid_kpt = np.where(np.any(np.isnan(keypoints), axis=1),
                         np.zeros_like(keypoints[:, 0]),
                         np.ones_like(keypoints[:, 0]))
  valid_kpt = [v for v in valid_kpt]

  for keypoint_x, keypoint_y, valid in zip(keypoints_x, keypoints_y, valid_kpt):
    if valid:
      draw.ellipse([(keypoint_x - radius, keypoint_y - radius),
                    (keypoint_x + radius, keypoint_y + radius)],
                   outline=color, fill=color)
  if keypoint_edges is not None:
    for keypoint_start, keypoint_end in keypoint_edges:
      if (keypoint_start < 0 or keypoint_start >= len(keypoints) or
          keypoint_end < 0 or keypoint_end >= len(keypoints)):
        continue
      if not (valid_kpt[keypoint_start] and valid_kpt[keypoint_end]):
        continue
      edge_coordinates = [
          keypoints_x[keypoint_start], keypoints_y[keypoint_start],
          keypoints_x[keypoint_end], keypoints_y[keypoint_end]
      ]
      draw.line(
          edge_coordinates, fill=keypoint_edge_color, width=keypoint_edge_width)


def draw_mask_on_image_array(image, mask, color='red', alpha=0.4):
 
  if image.dtype != np.uint8:
    raise ValueError('`image` not of type np.uint8')
  if mask.dtype != np.uint8:
    raise ValueError('`mask` not of type np.uint8')
  if image.shape[:2] != mask.shape:
    raise ValueError('The image has spatial dimensions %s but the mask has '
                     'dimensions %s' % (image.shape[:2], mask.shape))
  rgb = ImageColor.getrgb(color)
  pil_image = Image.fromarray(image)

  solid_color = np.expand_dims(
      np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
  pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
  pil_mask = Image.fromarray(np.uint8(255.0*alpha*(mask > 0))).convert('L')
  pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
  np.copyto(image, np.array(pil_image.convert('RGB')))


def draw_part_mask_on_image_array(image, mask, alpha=0.4, num_parts=24):
 
  if image.dtype != np.uint8:
    raise ValueError('`image` not of type np.uint8')
  if mask.dtype != np.uint8:
    raise ValueError('`mask` not of type np.uint8')
  if image.shape[:2] != mask.shape:
    raise ValueError('The image has spatial dimensions %s but the mask has '
                     'dimensions %s' % (image.shape[:2], mask.shape))

  pil_image = Image.fromarray(image)
  part_colors = np.zeros_like(image)
  mask_1_channel = mask[:, :, np.newaxis]
  for i, color in enumerate(STANDARD_COLORS[:num_parts]):
    rgb = np.array(ImageColor.getrgb(color), dtype=np.uint8)
    part_colors += (mask_1_channel == i + 1) * rgb[np.newaxis, np.newaxis, :]
  pil_part_colors = Image.fromarray(np.uint8(part_colors)).convert('RGBA')
  pil_mask = Image.fromarray(np.uint8(255.0 * alpha * (mask > 0))).convert('L')
  pil_image = Image.composite(pil_part_colors, pil_image, pil_mask)
  np.copyto(image, np.array(pil_image.convert('RGB')))


def draw_float_channel_on_image_array(image, channel, mask, alpha=0.9,
                                      cmap='YlGn'):

  if image.dtype != np.uint8:
    raise ValueError('`image` not of type np.uint8')
  if channel.dtype != np.float32:
    raise ValueError('`channel` not of type np.float32')
  if mask.dtype != np.uint8:
    raise ValueError('`mask` not of type np.uint8')
  if image.shape[:2] != channel.shape:
    raise ValueError('The image has spatial dimensions %s but the channel has '
                     'dimensions %s' % (image.shape[:2], channel.shape))
  if image.shape[:2] != mask.shape:
    raise ValueError('The image has spatial dimensions %s but the mask has '
                     'dimensions %s' % (image.shape[:2], mask.shape))

  cm = plt.get_cmap(cmap)
  pil_image = Image.fromarray(image)
  colored_channel = cm(channel)[:, :, :3]
  pil_colored_channel = Image.fromarray(
      np.uint8(colored_channel * 255)).convert('RGBA')
  pil_mask = Image.fromarray(np.uint8(255.0 * alpha * (mask > 0))).convert('L')
  pil_image = Image.composite(pil_colored_channel, pil_image, pil_mask)
  np.copyto(image, np.array(pil_image.convert('RGB')))


def visualize_boxes_and_labels_on_image_array(
    image,
    boxes,
    classes,
    scores,
    category_index,
    instance_masks=None,
    instance_boundaries=None,
    keypoints=None,
    keypoint_scores=None,
    keypoint_edges=None,
    track_ids=None,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=.5,
    agnostic_mode=False,
    line_thickness=4,
    groundtruth_box_visualization_color='black',
    skip_boxes=False,
    skip_scores=False,
    skip_labels=False,
    skip_track_ids=False):
 
  box_to_display_str_map = collections.defaultdict(list)
  box_to_color_map = collections.defaultdict(str)
  box_to_instance_masks_map = {}
  box_to_instance_boundaries_map = {}
  box_to_keypoints_map = collections.defaultdict(list)
  box_to_keypoint_scores_map = collections.defaultdict(list)
  box_to_track_ids_map = {}
  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
  for i in range(boxes.shape[0]):
    if max_boxes_to_draw == len(box_to_color_map):
      break
    if scores is None or scores[i] > min_score_thresh:
      box = tuple(boxes[i].tolist())
      if instance_masks is not None:
        box_to_instance_masks_map[box] = instance_masks[i]
      if instance_boundaries is not None:
        box_to_instance_boundaries_map[box] = instance_boundaries[i]
      if keypoints is not None:
        box_to_keypoints_map[box].extend(keypoints[i])
      if keypoint_scores is not None:
        box_to_keypoint_scores_map[box].extend(keypoint_scores[i])
      if track_ids is not None:
        box_to_track_ids_map[box] = track_ids[i]
      if scores is None:
        box_to_color_map[box] = groundtruth_box_visualization_color
      else:
        display_str = ''
        if not skip_labels:
          if not agnostic_mode:
            if classes[i] in six.viewkeys(category_index):
              class_name = category_index[classes[i]]['name']
            else:
              class_name = 'N/A'
            display_str = str(class_name)
        if not skip_scores:
          if not display_str:
            display_str = '{}%'.format(round(100*scores[i]))
          else:
            display_str = '{}: {}%'.format(display_str, round(100*scores[i]))
        if not skip_track_ids and track_ids is not None:
          if not display_str:
            display_str = 'ID {}'.format(track_ids[i])
          else:
            display_str = '{}: ID {}'.format(display_str, track_ids[i])
        box_to_display_str_map[box].append(display_str)
        if agnostic_mode:
          box_to_color_map[box] = 'DarkOrange'
        elif track_ids is not None:
          prime_multipler = _get_multiplier_for_color_randomness()
          box_to_color_map[box] = STANDARD_COLORS[
              (prime_multipler * track_ids[i]) % len(STANDARD_COLORS)]
        else:
          box_to_color_map[box] = STANDARD_COLORS[
              classes[i] % len(STANDARD_COLORS)]

  # Draw all boxes onto image.
  for box, color in box_to_color_map.items():
    ymin, xmin, ymax, xmax = box
    if instance_masks is not None:
      draw_mask_on_image_array(
          image,
          box_to_instance_masks_map[box],
          color=color
      )
    if instance_boundaries is not None:
      draw_mask_on_image_array(
          image,
          box_to_instance_boundaries_map[box],
          color='red',
          alpha=1.0
      )
    draw_bounding_box_on_image_array(
        image,
        ymin,
        xmin,
        ymax,
        xmax,
        color=color,
        thickness=0 if skip_boxes else line_thickness,
        display_str_list=box_to_display_str_map[box],
        use_normalized_coordinates=use_normalized_coordinates)
    if keypoints is not None:
      keypoint_scores_for_box = None
      if box_to_keypoint_scores_map:
        keypoint_scores_for_box = box_to_keypoint_scores_map[box]
      draw_keypoints_on_image_array(
          image,
          box_to_keypoints_map[box],
          keypoint_scores_for_box,
          min_score_thresh=min_score_thresh,
          color=color,
          radius=line_thickness / 2,
          use_normalized_coordinates=use_normalized_coordinates,
          keypoint_edges=keypoint_edges,
          keypoint_edge_color=color,
          keypoint_edge_width=line_thickness // 2)

  return image


def add_cdf_image_summary(values, name):
 
  def cdf_plot(values):
    """Numpy function to plot CDF."""
    normalized_values = values / np.sum(values)
    sorted_values = np.sort(normalized_values)
    cumulative_values = np.cumsum(sorted_values)
    fraction_of_examples = (np.arange(cumulative_values.size, dtype=np.float32)
                            / cumulative_values.size)
    fig = plt.figure(frameon=False)
    ax = fig.add_subplot('111')
    ax.plot(fraction_of_examples, cumulative_values)
    ax.set_ylabel('cumulative normalized values')
    ax.set_xlabel('fraction of examples')
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(
        1, int(height), int(width), 3)
    return image
  cdf_plot = tf.py_func(cdf_plot, [values], tf.uint8)
  tf.summary.image(name, cdf_plot)


def add_hist_image_summary(values, bins, name):
   def hist_plot(values, bins):
    """Numpy function to plot hist."""
    fig = plt.figure(frameon=False)
    ax = fig.add_subplot('111')
    y, x = np.histogram(values, bins=bins)
    ax.plot(x[:-1], y)
    ax.set_ylabel('count')
    ax.set_xlabel('value')
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.fromstring(
        fig.canvas.tostring_rgb(), dtype='uint8').reshape(
            1, int(height), int(width), 3)
    return image
  hist_plot = tf.py_func(hist_plot, [values, bins], tf.uint8)
  tf.summary.image(name, hist_plot)


class EvalMetricOpsVisualization(six.with_metaclass(abc.ABCMeta, object)):
  
  def __init__(self,
               category_index,
               max_examples_to_draw=5,
               max_boxes_to_draw=20,
               min_score_thresh=0.2,
               use_normalized_coordinates=True,
               summary_name_prefix='evaluation_image',
               keypoint_edges=None):
       self._category_index = category_index
    self._max_examples_to_draw = max_examples_to_draw
    self._max_boxes_to_draw = max_boxes_to_draw
    self._min_score_thresh = min_score_thresh
    self._use_normalized_coordinates = use_normalized_coordinates
    self._summary_name_prefix = summary_name_prefix
    self._keypoint_edges = keypoint_edges
    self._images = []

  def clear(self):
    self._images = []

  def add_images(self, images):
    """Store a list of images, each with shape [1, H, W, C]."""
    if len(self._images) >= self._max_examples_to_draw:
      return

    # Store images and clip list if necessary.
    self._images.extend(images)
    if len(self._images) > self._max_examples_to_draw:
      self._images[self._max_examples_to_draw:] = []

  def get_estimator_eval_metric_ops(self, eval_dict):
    
    if self._max_examples_to_draw == 0:
      return {}
    images = self.images_from_evaluation_dict(eval_dict)

    def get_images():
      """Returns a list of images, padded to self._max_images_to_draw."""
      images = self._images
      while len(images) < self._max_examples_to_draw:
        images.append(np.array(0, dtype=np.uint8))
      self.clear()
      return images

    def image_summary_or_default_string(summary_name, image):
      """Returns image summaries for non-padded elements."""
      return tf.cond(
          tf.equal(tf.size(tf.shape(image)), 4),
          lambda: tf.summary.image(summary_name, image),
          lambda: tf.constant(''))

    if tf.executing_eagerly():
      update_op = self.add_images([[images[0]]])
      image_tensors = get_images()
    else:
      update_op = tf.py_func(self.add_images, [[images[0]]], [])
      image_tensors = tf.py_func(
          get_images, [], [tf.uint8] * self._max_examples_to_draw)
    eval_metric_ops = {}
    for i, image in enumerate(image_tensors):
      summary_name = self._summary_name_prefix + '/' + str(i)
      value_op = image_summary_or_default_string(summary_name, image)
      eval_metric_ops[summary_name] = (value_op, update_op)
    return eval_metric_ops

  @abc.abstractmethod
  def images_from_evaluation_dict(self, eval_dict):
    """Converts evaluation dictionary into a list of image tensors.

    To be overridden by implementations.

    Args:
      eval_dict: A dictionary with all the necessary information for producing
        visualizations.

    Returns:
      A list of [1, H, W, C] uint8 tensors.
    """
    raise NotImplementedError


class VisualizeSingleFrameDetections(EvalMetricOpsVisualization):
  """Class responsible for single-frame object detection visualizations."""

  def __init__(self,
               category_index,
               max_examples_to_draw=5,
               max_boxes_to_draw=20,
               min_score_thresh=0.2,
               use_normalized_coordinates=True,
               summary_name_prefix='Detections_Left_Groundtruth_Right',
               keypoint_edges=None):
    super(VisualizeSingleFrameDetections, self).__init__(
        category_index=category_index,
        max_examples_to_draw=max_examples_to_draw,
        max_boxes_to_draw=max_boxes_to_draw,
        min_score_thresh=min_score_thresh,
        use_normalized_coordinates=use_normalized_coordinates,
        summary_name_prefix=summary_name_prefix,
        keypoint_edges=keypoint_edges)

  def images_from_evaluation_dict(self, eval_dict):
    return draw_side_by_side_evaluation_image(eval_dict, self._category_index,
                                              self._max_boxes_to_draw,
                                              self._min_score_thresh,
                                              self._use_normalized_coordinates,
                                              self._keypoint_edges)
