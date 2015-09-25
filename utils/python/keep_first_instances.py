#!/usr/bin/env python3
import os
import os.path
import sys

top_n = {'test':20,'train':80}
rootdir = sys.argv[1]
classes = [s for s in os.listdir(rootdir) if os.path.isdir(os.path.join(rootdir,s))]
startidx = {}
startidx['test'] = {'cone': 168, 'cup': 80, 'tent': 164, 'plant': 241, 'bookshelf': 573,
            'toilet': 345, 'dresser': 201, 'bottle': 336, 'laptop': 150, 'stairs': 125,
            'night_stand': 201, 'sofa': 681, 'stool': 91, 'car': 198, 'chair': 890,
            'piano': 232, 'flower_pot': 150, 'bathtub': 107, 'bench': 174, 'table': 393,
            'airplane': 627, 'monitor': 466, 'vase': 476, 'door': 110, 'guitar': 156,
            'lamp': 125, 'glass_box': 172, 'tv_stand': 268, 'person': 89, 'keyboard': 146,
            'curtain': 139, 'mantel': 285, 'bed': 516, 'desk': 201, 'wardrobe': 88,
            'sink': 129, 'radio': 105, 'bowl': 65, 'range_hood': 116, 'xbox': 104}
startidx['train'] = {'cone': 1, 'cup': 1, 'tent': 1, 'plant': 1, 'bookshelf': 1,
                     'toilet': 1, 'dresser': 1, 'bottle': 1, 'laptop': 1, 'stairs': 1,
                     'night_stand': 1, 'sofa': 1, 'stool': 1, 'car': 1, 'chair': 1,
                     'piano': 1, 'flower_pot': 1, 'bathtub': 1, 'bench': 1, 'table': 1,
                     'airplane': 1, 'monitor': 1, 'vase': 1, 'door': 1, 'guitar': 1,
                     'lamp': 1, 'glass_box': 1, 'tv_stand': 1, 'person': 1, 'keyboard': 1,
                     'curtain': 1, 'mantel': 1, 'bed': 1, 'desk': 1, 'wardrobe': 1,
                     'sink': 1, 'radio': 1, 'bowl': 1, 'range_hood': 1, 'xbox': 1}

for (i,c) in enumerate(classes):
    for s in top_n.keys():
        curr_path = os.path.join(rootdir,c,s)
        files = os.listdir(curr_path)
        files.sort()
        files_to_rm = files[top_n[s]:]
        files_to_keep = files[:top_n[s]]
        for f in files_to_rm:
            os.remove(os.path.join(rootdir,c,s,f))
        for (idx,f) in enumerate(files_to_keep):
            f_new = '{0}_{1:0>4}.off'.format(c,idx+startidx[s][c])
            os.rename(os.path.join(rootdir,c,s,f),os.path.join(rootdir,c,s,f_new))
        print('[{0}/{1}]{2}/{3}: {4} files removed.'.format(i+1,len(classes),c,s,len(files_to_rm)))
