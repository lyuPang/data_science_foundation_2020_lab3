from cv2 import cv2 
import selectivesearch


def sift(img_path):
    img=cv2.imread(img_path)
    _sift = cv2.xfeatures2d.SIFT_create()
    key_point,descriptors=_sift.detectAndCompute(img,None)
    print(len(key_point))
    print(descriptors.shape)
    return key_point,descriptors


def selective_search(img_path):
    # https://github.com/AlpacaDB/selectivesearch
    img=cv2.imread(img_path)
    img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)
    # 创建一个集合 元素不会重复，每一个元素都是一个list(左上角x，左上角y,宽,高)，表示一个候选区域的边框
    candidates = set()
    for r in regions:
        # 排除重复的候选区
        if r['rect'] in candidates:
            continue
        # 排除小于 2000 pixels的候选区域(并不是bounding box中的区域大小)
        if r['size'] < 2000:
            continue
        # 排除扭曲的候选区域边框  即只保留近似正方形的
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])
    img_regions=[]
    for candidate in candidates:
        x, y, w, h = candidate
        img_regions.append(img[x:x+w][y:y+h])
    return img_regions



if __name__=='__main__':
    img='1.jpg'
    print(sift(img))
    print(selective_search(img))
