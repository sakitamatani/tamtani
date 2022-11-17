from enum import Enum

EVAL_EXCESS_DIR = 'eval_excess'
EVAL_DEFICIENCY_DIR = 'eval_deficiency'
EVAL_EXCESS_AND_DEFICIENCY_DIR = 'eval_excess_and_deficiency'

CLASS_NAME_PERSON = 'Person'
CLASS_NAME_CAT = 'Cat'
CLASS_NAME_DOG = 'Dog'
CLASS_NAMES = [CLASS_NAME_PERSON, CLASS_NAME_CAT, CLASS_NAME_DOG]


class OutputType(Enum):
    TEXT = 0
    IMAGE = 1
    BOTH = 2


def prepare_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def eval_excesses_and_deficiencies(result_name, data_type,output_type=OutputType.TEXT, with_save=False):
    pred_mask_dir = RESULTS_DIR(result_name, DATATYPE_PREDICTS(data_type), RESULT_TYPE_MASK)
    teacher_mask_dir = RESULTS_DIR(result_name, DATATYPE_TEACHERS(data_type), RESULT_TYPE_MASK)

    print('target pred dir : ', pred_mask_dir)

    pred_mask_files = glob.glob(os.path.join(pred_mask_dir, '*.png'))
    pred_mask_files.sort()

    all_excesses = {}
    all_deficiencies = {}
    all_deficiencies_to_teacher = {}
    count_teachers = {}

    for target_idx in IDXES:
        all_excesses[target_idx] = []
        all_deficiencies[target_idx] = []
        all_deficiencies_to_teacher[target_idx] = []
        count_teachers[target_idx] = 0


    if len(pred_mask_files) < 1:
        print('pred_mask_files is []')

    for k, p_mask_file in enumerate(pred_mask_files):
        fname = os.path.basename(p_mask_file)
        t_mask_file = os.path.join(teacher_mask_dir, fname)
        if not os.path.exists(t_mask_file):
            print('teacher mask is not found.')
            continue

        p_masks = cv2.imread(p_mask_file)
        t_masks = cv2.imread(t_mask_file)

        save_base_dir = RESULTS_DIR(result_name, model_type, data_type, '')
        eval_excess_and_deficiency(fname, p_masks, t_masks, all_excesses, all_deficiencies,
                    all_deficiencies_to_teacher, count_teachers,
                    output_type=output_type, with_save=with_save, save_base_dir=save_base_dir)

    if output_type == OutputType.TEXT or output_type == OutputType.BOTH:
        mean_excesses_print = ':'
        mean_deficiencies_print = ':'
        mean_deficiencies_to_teacher_print = ':'
        excesses_rate_print = ':'
        deficiencies_rate_print = ':'
        deficiencies_to_teacher_rate_print = ':'
        for target_idx in IDXES:
            mean_excesses = np.average(np.array(all_excesses[target_idx]))
            mean_deficiencies = np.average(np.array(all_deficiencies[target_idx]))
            mean_deficiencies_to_teacher = np.average(np.array(all_deficiencies_to_teacher[target_idx]))
            mean_excesses_print += '%.3f, ' % mean_excesses
            mean_deficiencies_print += '%.3f, ' % mean_deficiencies
            mean_deficiencies_to_teacher_print += '%.3f, ' % mean_deficiencies_to_teacher
            excesses_rate_print += '%.3f, ' % (len(all_excesses[target_idx]) / len(pred_mask_files))
            deficiencies_rate_print += '%.3f, ' % (len(all_deficiencies[target_idx]) / len(pred_mask_files))
            deficiencies_to_teacher_rate = 0
            if count_teachers[target_idx] != 0:
                deficiencies_to_teacher_rate = len(all_deficiencies[target_idx]) / count_teachers[target_idx]
            deficiencies_to_teacher_rate_print += '%.3f, ' % deficiencies_to_teacher_rate
        print('')
        print('rate:', excesses_rate_print, deficiencies_rate_print)
        print('mean:', mean_excesses_print, mean_deficiencies_print, mean_deficiencies_to_teacher_print)


def eval_excess_and_deficiency(fname, p_masks, t_masks, all_excesses, all_deficiencies,
                            all_deficiencies_to_teacher, count_teachers,
                            output_type=OutputType.TEXT, with_save=False, save_base_dir=''):
    img_h, img_w = np.shape(p_masks)[:2]
    img_size = img_h * img_w

    excesses_print = ':'
    deficiencies_print = ':'
    deficiencies_to_teacher_print = ':'
    image_plotter = ImageLinePlotter(plot_area_num=len(IDXES))
    for target_idx, cname in zip(IDXES, CLASS_NAMES):
        p_mask = to_binary(p_masks[:, :, target_idx])
        t_mask = to_binary(t_masks[:, :, target_idx])
        e_mask, d_mask, o_mask = make_excess_and_deficiency(p_mask, t_mask)
        excesses = np.sum(e_mask) / img_size
        deficiencies = np.sum(d_mask) / img_size
        if np.sum(t_mask) != 0:
            count_teachers[target_idx] += 1
            deficiencies_to_teacher = np.sum(d_mask) / np.sum(t_mask)
        else:
            deficiencies_to_teacher = 0
        excesses_print += '%.3f, ' % excesses
        deficiencies_print += '%.3f, ' % deficiencies
        deficiencies_to_teacher_print += '%.3f, ' % deficiencies_to_teacher
        if excesses != 0:
            all_excesses[target_idx].append(excesses)
        if deficiencies != 0:
            all_deficiencies[target_idx].append(deficiencies)
        if deficiencies_to_teacher != 0:
            all_deficiencies_to_teacher[target_idx].append(deficiencies_to_teacher)
        dummy_mask = np.zeros(np.shape(p_mask)[:2], dtype=np.uint8)
        eval_img = np.stack([d_mask * 255, o_mask * 120, e_mask * 255], axis=2)
        eval_img = eval_img.astype(np.uint8)
        image_plotter.add_image(eval_img, title=cname)
        if with_save:
            e_img = np.stack([dummy_mask, dummy_mask, e_mask * 255], axis=2)
            d_img = np.stack([d_mask * 255, dummy_mask, dummy_mask], axis=2)
            save_excess_dir = prepare_dir(os.path.join(save_base_dir, EVAL_EXCESS_DIR))
            save_deficiency_dir = prepare_dir(os.path.join(save_base_dir, EVAL_DEFICIENCY_DIR))
            save_excess_and_deficiency_dir = prepare_dir(os.path.join(save_base_dir, EVAL_EXCESS_AND_DEFICIENCY_DIR))
            cv2.imwrite(os.path.join(save_excess_dir, fname), e_img)
            cv2.imwrite(os.path.join(save_deficiency_dir, fname), d_img)
            cv2.imwrite(os.path.join(save_excess_and_deficiency_dir, fname), eval_img)

    if output_type == OutputType.TEXT or output_type == OutputType.BOTH:
        print(fname, excesses_print, deficiencies_print, deficiencies_to_teacher_print)
    else:
        print(fname)

    if output_type == OutputType.IMAGE or output_type == OutputType.BOTH:
        image_plotter.show_plot()

    return all_excesses, all_deficiencies, all_deficiencies_to_teacher, count_teachers


def make_excess_and_deficiency(p_mask, t_mask):
    e_mask = (p_mask - t_mask).astype(np.int8)
    e_mask[e_mask < 0] = 0
    e_mask = e_mask.astype(np.uint8)

    d_mask = (t_mask - p_mask).astype(np.int8)
    d_mask[d_mask < 0] = 0
    d_mask = d_mask.astype(np.uint8)

    o_mask = p_mask * t_mask

    return e_mask, d_mask, o_mask
