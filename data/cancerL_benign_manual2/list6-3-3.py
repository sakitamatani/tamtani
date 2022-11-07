chk_data = os.path.join('data', 'train', 'cat_dog_', '2007_001763.png')

img = Image.open(chk_data)

img_arr = np.array(img)
print('original unique : ', np.unique(img_arr))
print('---------------------------------------------------')

pil_resize = img.resize((1000, 1000))
pil_resize_arr = np.array(pil_resize)
print('PIL resized unique : ', np.unique(pil_resize_arr))
print('---------------------------------------------------')

cv2_resize = cv2.resize(img_arr, (1000, 1000))
print('cv2 resized unique : ', np.unique(cv2_resize))
