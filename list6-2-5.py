import shutil

TARGET_DIR = os.path.join('.', 'target_data')

ID_PERSON = 15
ID_CAT = 8
ID_DOG = 12

cls_counters = {}
for seg_file in segs:
    seg_img_PIL = np.array(Image.open(seg_file))

    copy_to = ''
    if np.any(seg_img_PIL == ID_PERSON):
         copy_to += 'person_'
    if np.any(seg_img_PIL == ID_CAT):
        copy_to += 'cat_'
    if np.any(seg_img_PIL == ID_DOG):
        copy_to += 'dog_'

    if copy_to == '':
        continue
    if not copy_to in cls_counters.keys():
            cls_counters[copy_to] = 0
    cls_counters[copy_to] += 1

    copy_path = os.path.join(TARGET_DIR, copy_to)
    if not os.path.exists(copy_path):
        os.makedirs(copy_path)
    shutil.copy(seg_file, copy_path)

print(cls_counters)
