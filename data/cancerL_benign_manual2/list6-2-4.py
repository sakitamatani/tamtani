ORG_DIR = os.path.join('.', 'VOC2012', 'JPEGImages')
SEG_DIR = os.path.join('.', 'VOC2012', 'SegmentationClass')

segs = glob.glob(os.path.join(SEG_DIR, '*.png'))
segs.sort()

for seg_file in segs[:5]:
    print('# ', os.path.basename(seg_file))
    fname, _ = os.path.splitext(os.path.basename(seg_file))
    org_file = os.path.join(ORG_DIR, fname + '.jpg')
    if not os.path.exists(org_file):
        print('not found org_file : ', org_file)
    org_img = cv2.imread(org_file)
    seg_img_PIL = np.array(Image.open(seg_file))
    seg_img_cv2 = cv2.imread(seg_file)

    print('seg_img_PIL values : ', np.unique(seg_img_PIL))
    print('seg_img_cv2 values : ', np.unique(seg_img_cv2))
