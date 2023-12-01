import numpy as np
import tensorflow as tf
import cv2
import visualization_utils as vis_util

def create_category_index(label_path='coco_ssd_mobilenet/labelmap.txt'):
   
    f = open(label_path)
    category_index = {}
    for i, val in enumerate(f):
        if i != 0:
            val = val[:-1]
            if val != '???':
                category_index.update({(i-1): {'id': (i-1), 'name': val}})
            
    f.close()
    return category_index
def get_output_dict(image, interpreter, output_details, nms=True, iou_thresh=0.5, score_thresh=0.6):
    
    output_dict = {
                   'detection_boxes' : interpreter.get_tensor(output_details[0]['index'])[0],
                   'detection_classes' : interpreter.get_tensor(output_details[1]['index'])[0],
                   'detection_scores' : interpreter.get_tensor(output_details[2]['index'])[0],
                   'num_detections' : interpreter.get_tensor(output_details[3]['index'])[0]
                   }

    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    if nms:
        output_dict = apply_nms(output_dict, iou_thresh, score_thresh)
    return output_dict

def apply_nms(output_dict, iou_thresh=0.5, score_thresh=0.6):
   
    q = 90 # no of classes
    num = int(output_dict['num_detections'])
    boxes = np.zeros([1, num, q, 4])
    scores = np.zeros([1, num, q])
    # val = [0]*q
    for i in range(num):
        # indices = np.where(classes == output_dict['detection_classes'][i])[0][0]
        boxes[0, i, output_dict['detection_classes'][i], :] = output_dict['detection_boxes'][i]
        scores[0, i, output_dict['detection_classes'][i]] = output_dict['detection_scores'][i]
    nmsd = tf.image.combined_non_max_suppression(boxes=boxes,
                                                 scores=scores,
                                                 max_output_size_per_class=num,
                                                 max_total_size=num,
                                                 iou_threshold=iou_thresh,
                                                 score_threshold=score_thresh,
                                                 pad_per_class=False,
                                                 clip_boxes=False)
    valid = nmsd.valid_detections[0].numpy()
    output_dict = {
                   'detection_boxes' : nmsd.nmsed_boxes[0].numpy()[:valid],
                   'detection_classes' : nmsd.nmsed_classes[0].numpy().astype(np.int64)[:valid],
                   'detection_scores' : nmsd.nmsed_scores[0].numpy()[:valid],
                   }
    return output_dict

def make_and_show_inference(img, interpreter, input_details, output_details, category_index, nms=True, score_thresh=0.6, iou_thresh=0.5):
   
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (300, 300), cv2.INTER_AREA)
    img_rgb = img_rgb.reshape([1, 300, 300, 3])

    interpreter.set_tensor(input_details[0]['index'], img_rgb)
    interpreter.invoke()
    
    output_dict = get_output_dict(img_rgb, interpreter, output_details, nms, iou_thresh, score_thresh)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
    img,
    output_dict['detection_boxes'],
    output_dict['detection_classes'],
    output_dict['detection_scores'],
    category_index,
    use_normalized_coordinates=True,
    min_score_thresh=score_thresh,
    line_thickness=3)

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="coco_ssd_mobilenet/detect.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

category_index = create_category_index()
input_shape = input_details[0]['shape']
cap = cv2.VideoCapture(0)

while(True):
    ret, img = cap.read()
    if ret:
        make_and_show_inference(img, interpreter, input_details, output_details, category_index)
        cv2.imshow("image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    
cap.release()
cv2.destroyAllWindows()
