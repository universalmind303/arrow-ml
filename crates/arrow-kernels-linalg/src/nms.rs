use arrow::array::Int64Array;
use arrow::datatypes::Float32Type;
use arrow::tensor::Tensor;
use arrow_kernels_common::{KernelError, Result};

/// Non-Maximum Suppression for object detection.
///
/// - `boxes`: f32 tensor of shape (num_boxes, 4) with [x1, y1, x2, y2] format
/// - `scores`: f32 tensor of shape (num_boxes,) with confidence scores
/// - `max_output_boxes`: maximum number of boxes to keep
/// - `iou_threshold`: IoU threshold for suppression (e.g. 0.5)
/// - `score_threshold`: minimum score to consider (boxes below are discarded)
///
/// Returns Int64Array of selected box indices.
pub fn non_max_suppression(
    boxes: &Tensor<'_, Float32Type>,
    scores: &Tensor<'_, Float32Type>,
    max_output_boxes: usize,
    iou_threshold: f32,
    score_threshold: f32,
) -> Result<Int64Array> {
    let box_shape = boxes
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("nms: boxes has no shape".into()))?;
    if box_shape.len() != 2 || box_shape[1] != 4 {
        return Err(KernelError::InvalidArgument(format!(
            "nms: boxes must be (N, 4), got {:?}",
            box_shape
        )));
    }
    let num_boxes = box_shape[0];

    let score_shape = scores
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("nms: scores has no shape".into()))?;
    if score_shape.len() != 1 || score_shape[0] != num_boxes {
        return Err(KernelError::ShapeMismatch {
            operation: "nms",
            expected: format!("scores shape [{num_boxes}]"),
            actual: format!("scores shape {:?}", score_shape),
        });
    }

    let box_data: &[f32] = boxes.data().typed_data();
    let score_data: &[f32] = scores.data().typed_data();

    // Sort indices by score descending
    let mut order: Vec<usize> = (0..num_boxes)
        .filter(|&i| score_data[i] >= score_threshold)
        .collect();
    order.sort_by(|&a, &b| {
        score_data[b]
            .partial_cmp(&score_data[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep = Vec::new();
    let mut suppressed = vec![false; num_boxes];

    for &i in &order {
        if suppressed[i] {
            continue;
        }
        keep.push(i as i64);
        if keep.len() >= max_output_boxes {
            break;
        }

        let (x1_i, y1_i, x2_i, y2_i) = (
            box_data[i * 4],
            box_data[i * 4 + 1],
            box_data[i * 4 + 2],
            box_data[i * 4 + 3],
        );
        let area_i = (x2_i - x1_i).max(0.0) * (y2_i - y1_i).max(0.0);

        for &j in &order {
            if suppressed[j] || j == i {
                continue;
            }
            let (x1_j, y1_j, x2_j, y2_j) = (
                box_data[j * 4],
                box_data[j * 4 + 1],
                box_data[j * 4 + 2],
                box_data[j * 4 + 3],
            );
            let area_j = (x2_j - x1_j).max(0.0) * (y2_j - y1_j).max(0.0);

            let inter_x1 = x1_i.max(x1_j);
            let inter_y1 = y1_i.max(y1_j);
            let inter_x2 = x2_i.min(x2_j);
            let inter_y2 = y2_i.min(y2_j);

            let inter_area = (inter_x2 - inter_x1).max(0.0) * (inter_y2 - inter_y1).max(0.0);
            let union_area = area_i + area_j - inter_area;

            let iou = if union_area > 0.0 {
                inter_area / union_area
            } else {
                0.0
            };
            if iou >= iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    Ok(Int64Array::from(keep))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::buffer::{Buffer, ScalarBuffer};

    fn make_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<'static, Float32Type> {
        let buffer = Buffer::from(ScalarBuffer::<f32>::from(data).into_inner());
        Tensor::new_row_major(buffer, Some(shape), None).unwrap()
    }

    #[test]
    fn test_nms_basic() {
        // Two overlapping boxes, one higher score
        let boxes = make_f32(
            vec![
                0.0, 0.0, 10.0, 10.0, // box 0
                1.0, 1.0, 11.0, 11.0, // box 1 (high overlap with box 0)
                20.0, 20.0, 30.0, 30.0, // box 2 (no overlap)
            ],
            vec![3, 4],
        );
        let scores = make_f32(vec![0.9, 0.8, 0.7], vec![3]);
        let result = non_max_suppression(&boxes, &scores, 10, 0.5, 0.0).unwrap();
        // Box 0 has highest score, suppresses box 1 (high IoU)
        // Box 2 survives (no overlap)
        assert_eq!(result.len(), 2);
        assert_eq!(result.value(0), 0); // highest score
        assert_eq!(result.value(1), 2); // non-overlapping
    }

    #[test]
    fn test_nms_max_output() {
        let boxes = make_f32(
            vec![
                0.0, 0.0, 1.0, 1.0, 10.0, 10.0, 11.0, 11.0, 20.0, 20.0, 21.0, 21.0,
            ],
            vec![3, 4],
        );
        let scores = make_f32(vec![0.9, 0.8, 0.7], vec![3]);
        let result = non_max_suppression(&boxes, &scores, 2, 0.5, 0.0).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_nms_score_threshold() {
        let boxes = make_f32(vec![0.0, 0.0, 1.0, 1.0, 10.0, 10.0, 11.0, 11.0], vec![2, 4]);
        let scores = make_f32(vec![0.9, 0.3], vec![2]);
        let result = non_max_suppression(&boxes, &scores, 10, 0.5, 0.5).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result.value(0), 0);
    }

    #[test]
    fn test_nms_no_overlap() {
        // All boxes far apart -> all kept
        let boxes = make_f32(
            vec![
                0.0, 0.0, 1.0, 1.0, 10.0, 10.0, 11.0, 11.0, 20.0, 20.0, 21.0, 21.0,
            ],
            vec![3, 4],
        );
        let scores = make_f32(vec![0.5, 0.6, 0.7], vec![3]);
        let result = non_max_suppression(&boxes, &scores, 10, 0.5, 0.0).unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_nms_invalid_boxes_shape() {
        let boxes = make_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let scores = make_f32(vec![0.5], vec![1]);
        assert!(non_max_suppression(&boxes, &scores, 10, 0.5, 0.0).is_err());
    }
}
