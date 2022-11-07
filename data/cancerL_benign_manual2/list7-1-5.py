RESULT_NAME_1 = 'results_001'

RESULT_TYPE_MASK = 'predict_outputs'
RESULT_TYPE_ARGMAX = 'argmax_outputs'
RESULT_TYPE_OVERLAY = 'overlay_outputs'

DATATYPE_TRAIN = 'train'
DATATYPE_VALID = 'valid'

def DATATYPE_PREDICTS(data_type):
    return data_type + '_predicts'

def DATATYPE_TEACHERS(data_type):
    return data_type + '_teachers'

IDX_CLASS_PERSON = 0
IDX_CLASS_CAT = 1
IDX_CLASS_DOG = 2
IDXES = [IDX_CLASS_PERSON, IDX_CLASS_CAT, IDX_CLASS_DOG]

def RESULTS_DIR(result_name, data_type, result_type):
    return os.path.join(result_name, data_type, result_type)


def eval_iou(result_name, data_type):
    pred_mask_dir = RESULTS_DIR(result_name, DATATYPE_PREDICTS(data_type), RESULT_TYPE_MASK)
    teacher_mask_dir = RESULTS_DIR(result_name, DATATYPE_TEACHERS(data_type), RESULT_TYPE_MASK)
    if not os.path.exists(teacher_mask_dir):
        print('teachers dir is not found. : ', teacher_mask_dir)
        return False

    print('target pred dir : ', pred_mask_dir)

    pred_mask_files = glob.glob(os.path.join(pred_mask_dir, '*.png'))
    pred_mask_files.sort()

    all_ious = {}
    for target_idx in IDXES:
        all_ious[target_idx] = []

    if len(pred_mask_files) < 1:
        print('pred_mask_files is []')

    for p_mask_file in pred_mask_files:
        fname = os.path.basename(p_mask_file)
        t_mask_file = os.path.join(teacher_mask_dir, fname)
        if not os.path.exists(t_mask_file):
            print('teacher mask is not found.')
            continue

        p_masks = cv2.imread(p_mask_file)
        t_masks = cv2.imread(t_mask_file)

        iou_print = ':'
        for target_idx in IDXES:
            p_mask = p_masks[:, :, target_idx]
            t_mask = t_masks[:, :, target_idx]
            iou = calculate_iou(p_mask, t_mask)
            iou_print += '%.3f, ' % iou
            if np.isnan(iou):
                continue
            all_ious[target_idx].append(iou)

        print(fname, iou_print)

    mean_iou_print = 'mean:'
    for target_idx in IDXES:
        mean_iou = np.average(np.array(all_ious[target_idx]))
        mean_iou_print += '%.3f, ' % mean_iou
    print('')
    print(mean_iou_print)


def calculate_iou(p_mask, t_mask):
    p_mask = to_binary(p_mask)
    t_mask = to_binary(t_mask)

    if np.sum(t_mask) == 0:
        return np.nan

    overlap = np.sum(p_mask * t_mask)
    union = np.sum(to_binary(p_mask + t_mask))

    return overlap / union


def to_binary(mask, val=1):
    mask[mask > 0] = val
    return mask
	