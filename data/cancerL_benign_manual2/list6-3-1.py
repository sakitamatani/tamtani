import csv

DATA_BASE_DIR = os.path.join('.', 'data')
TRAIN_BASE_DIR = os.path.join(DATA_BASE_DIR, 'train')
VALID_BASE_DIR = os.path.join(DATA_BASE_DIR, 'valid')

VALID_FILES_CSV = os.path.join('.', '6-3-1_valid.csv')

reader = csv.reader(open(VALID_FILES_CSV))
valid_files = np.array(list(reader))[:, 0]

for d in os.listdir(TARGET_DIR):
    if not os.path.isdir(os.path.join(TARGET_DIR, d)):
        continue
    train_dir = os.path.join(TRAIN_BASE_DIR, d)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    valid_dir = os.path.join(VALID_BASE_DIR, d)
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)

    files = glob.glob(os.path.join(TARGET_DIR, d, '*.png'))
    for f in files:
        move_to_dir = train_dir
        if os.path.basename(f) in valid_files:
            move_to_dir = valid_dir
        shutil.move(f, move_to_dir)

print('all done.')
