# *_*coding:utf-8 *_*
import argparse
import json
import numpy as np


def parse_args():
    # Parse input arguments
    parser = argparse.ArgumentParser(description='filter out the classes we want from coco annotation')
    parser.add_argument('--evl_dir', dest='evl_dir',
                        help='annotation (evaluation) from dir',
                        type=str, default="C:/Users/Cambridge/Desktop/coco_instances_results.json")
    parser.add_argument('--gt_dir', dest='gt_dir',
                        help='annotation (ground truth) from dir',
                        type=str,
                        default='/media/shared-corpus/TableBank_data/Detection_data/Word/Word.json')
    args = parser.parse_args()
    return args


def iou_area(evl_patch, gt_patch):
    DetectedArea, GTArea, IOU = 0, 0, 0
    for h in range(len(evl_patch)):
        for i in range(len(gt_patch[h])):
            gt_x, gt_y, gt_w, gt_h = gt_patch[h][i]
            for j in range(len(evl_patch[h])):
                evl_x, evl_y, evl_w, evl_h = evl_patch[h][j]
                left = max(evl_x, gt_x)
                top = max(evl_y, gt_y)
                right = min(evl_x + evl_w, gt_x + gt_w)
                bottom = min(evl_y + evl_h, gt_y + gt_h)
                if left >= right or top >= bottom:
                    IOU += 0
                else:
                    inter = (right - left) * (bottom - top)
                    IOU += inter


        for i in range(len(gt_patch[h])):
            GTArea += gt_patch[h][i][2] * gt_patch[h][i][3]
        for i in range(len(evl_patch[h])):
            DetectedArea += evl_patch[h][i][2] * evl_patch[h][i][3]

    return DetectedArea, GTArea, IOU

# def inters(b1, b2):
#     gt_x, gt_y, gt_w, gt_h = b1
#     evl_x, evl_y, evl_w, evl_h = b2
#     left = max(evl_x, gt_x)
#     top = max(evl_y, gt_y)
#     right = min(evl_x + evl_w, gt_x + gt_w)
#     bottom = min(evl_y + evl_h, gt_y + gt_h)
#     if left >= right or top >= bottom:
#         return False
#     else:
#         return True

def suppression(patch_perimg, thresh=0):
    # 首先数据赋值和计算对应矩形框的面积
    dets = []
    for i in range(len(patch_perimg)):
        xmin, ymin, w, h = patch_perimg[i]['bbox']
        xmax, ymax = xmin+w, ymin+h
        scores = patch_perimg[i]['score']
        dets.append([xmin,ymin,xmax,ymax,scores])
    # dets的数据格式是dets[[xmin,ymin,xmax,ymax,scores]....]

    dets = np.array(dets)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    # print('areas  ', areas)
    # print('scores ', scores)

    # 这边的keep用于存放，NMS后剩余的方框
    keep = []

    # 取出分数从大到小排列的索引。.argsort()是从小到大排列，[::-1]是列表头和尾颠倒一下。
    index = scores.argsort()[::-1]
    # print(index)
    # 上面这两句比如分数[0.72 0.8  0.92 0.72 0.81 0.9 ]
    #  对应的索引index[  2   5    4     1    3   0  ]记住是取出索引，scores列表没变。

    # index会剔除遍历过的方框，和合并过的方框。
    while index.size > 0:
        # print(index.size)
        # 取出第一个方框进行和其他方框比对，看有没有可以合并的
        i = index[0]  # every time the first is the biggst, and add it directly

        # 因为我们这边分数已经按从大到小排列了。
        # 所以如果有合并存在，也是保留分数最高的这个，也就是我们现在那个这个
        # keep保留的是索引值，不是具体的分数。
        keep.append(i)
        # print(keep)
        # print('x1', x1[i])
        # print(x1[index[1:]])

        # 计算交集的左上角和右下角
        # 这里要注意，比如x1[i]这个方框的左上角x和所有其他的方框的左上角x的
        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        # print(x11, y11, x22, y22)
        # 这边要注意，如果两个方框相交，X22-X11和Y22-Y11是正的。
        # 如果两个方框不相交，X22-X11和Y22-Y11是负的，我们把不相交的W和H设为0.
        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)

        # 计算重叠面积就是上面说的交集面积。不相交因为W和H都是0，所以不相交面积为0
        overlaps = w * h
        # print('overlaps is', overlaps)

        # 这个就是IOU公式（交并比）。
        # 得出来的ious是一个列表，里面拥有当前方框和其他所有方框的IOU结果。
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        # print('ious is', ious)

        # 接下来是合并重叠度最大的方框，也就是合并ious中值大于thresh的方框
        # 我们合并的操作就是把他们剔除，因为我们合并这些方框只保留下分数最高的。
        # 我们经过排序当前我们操作的方框就是分数最高的，所以我们剔除其他和当前重叠度最高的方框
        # 这里np.where(ious<=thresh)[0]是一个固定写法。
        idx = np.where(ious <= thresh)[0]
        # print(idx)

        # 把留下来框在进行NMS操作
        # 这边留下的框是去除当前操作的框，和当前操作的框重叠度大于thresh的框
        # 每一次都会先去除当前操作框，所以索引的列表就会向前移动移位，要还原就+1，向后移动一位
        index = index[idx + 1]  # because index start from 1
        # print(index)
        patch_img = dets[keep].tolist()
        patch = []
        for i in range(len(patch_img)):
            left, top, right, botton, score = patch_img[i]
            patch.append([left, top, right-left, botton-top])

    return patch

if __name__ == "__main__":
    args = parse_args()
    # annot = json.load(open("E:/Cambridge/CambridgeCEngineering/PycharmProjects/PubLayNet-master/val.json", 'r'))
    evl = json.load(open(args.evl_dir, 'r'))
    gt = json.load(open(args.gt_dir, 'r'))

    DetectedArea = 0
    GTArea = 0
    IOU = 0

    j_start = 0
    k_start = 0
    evl_patch = []
    gt_patch = []
    for i in range(len(gt['images'])):
        # make patch of anns according to the image_id
        evl_patch_perimg = []
        for j in range(j_start, len(evl)):
            if evl[j]['image_id'] == gt['images'][i]['id']:
                evl_patch_perimg.append(evl[j])
            else:
                j_start = j
                break

        gt_patch_perimg = []
        for k in range(k_start, len(gt['annotations'])):
            if gt['annotations'][k]['image_id'] == gt['images'][i]['id']:
                gt_patch_perimg.append(gt['annotations'][k]['bbox'])
            else:
                k_start = k
                break
        if len(evl_patch_perimg)>1:
            evl_patch_perimg = suppression(evl_patch_perimg)
        elif len(evl_patch_perimg)==1:
            evl_patch_perimg = [evl_patch_perimg[0]['bbox']]
        evl_patch.append(evl_patch_perimg)
        gt_patch.append(gt_patch_perimg)

    DetectedArea, GTArea, IOU = iou_area(evl_patch, gt_patch)
    print(" DetectedArea = %f  " % DetectedArea)
    print(" GTArea       = %f  " % GTArea)
    print(" inter        = %f  " % IOU)
    print(" Precision    = %f  " % (IOU / DetectedArea))
    print(" Recall       = %f  " % (IOU / GTArea))

    print('---Finish---')
