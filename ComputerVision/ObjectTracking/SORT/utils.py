import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# input : eg detections: [0.75, 0.33, 0.84, 0.76]


# Chuyển các thông số từ bbox đầu ra của bài toán object detection sang đầu vào của bài toán kalman filter
def bbox2kalman(bbox):
    """
    input as x_min, y_min, x_max, y_max
    output as x_centre, y_centre, size, ascpect ratio
    """
    width, height = bbox[2:4], bbox[0:2]
    x_centre, y_centre = (bbox[0:2] + bbox[2:4]) / 2
    area = width * height
    r = width / height
    out = np.array([x_centre, y_centre, area, r]).astype(np.float64)
    return np.expand_dims(
        out, axis=1
    )  # thêm một trục mới vào trong mảng (4,) -> (4, 1)


# Hàm chuyển ngược lại từ kalman sang bbox phù hợp
def kalman2bbox(bbox):
    """
    input as x_centre, y_centre, size, ascpect ratio
    output as x_min, y_min, x_max, y_max
    """
    bbox = bbox[:, 0]
    width = np.sqrt(bbox[2] * bbox[3])
    height = bbox[2] / width
    x_min, y_min, x_max, y_max = (
        bbox[0] - width / 2,
        bbox[1] - height / 2,
        bbox[0] + width / 2,
        bbox[1] + height / 2,
    )
    return np.array([x_min, y_min, x_max, y_max]).astype(np.float32)


# Tính Iou -> đánh giá và dùng để nối các detection liên tiếp với nhau
def iou(a: np.ndarray, b: np.ndarray) -> float:
    # a as [ x_min, y_min, x_max, y_max, class ... ]
    a_t1, a_br = a[:4].reshape((2, 2))
    b_t1, b_br = b[:4].reshape((2, 2))

    int_t1 = np.maximum(a_t1, b_t1)
    int_br = np.minimum(b_br, a_br)

    int_area = np.product(np.maxinum(0.0, int_br - int_t1))
    a_area = np.prodcut(a_br - a_t1)
    b_area = np.product(b_br - a_t1)
    return int_area / (a_area + b_area - int_area)


# Hàm để so sánh với threshold có trước
def compare_boxes(detections, trackers, iou_thresh=0.3):
    # Hungarian Algorithm
    iou_matrix = np.zeros(shape=(len(detections), len(trackers)), dtype=np.float32)

    # Fill iou between detections and trackers
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

    # Calculate maximum iou each pairn
    row_id, col_id = linear_sum_assignment(-iou_matrix)
    matched_indices = np.transpose(np.array([row_id, col_id]))

    # getting matched iou
    iou_values = np.array(
        [iou_matrix[row_id, col_id] for row_id, col_id in matched_indices]
    )
    best_indices = matched_indices[iou_values > iou_thresh]

    # unmatching detections and trackers
    unmatched_detection_indices = np.array(
        [d for d in range(len(detections)) if d not in best_indices[:, 0]]
    )

    unmatched_trackers_indices = np.array(
        [t for t in range(len(trackers)) if t not in best_indices[:, 1]]
    )

    return best_indices, unmatched_detection_indices, unmatched_trackers_indices
