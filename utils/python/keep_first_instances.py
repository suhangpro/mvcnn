import os
import os.path
import sys

top_n = {'test':20,'train':80}
#rootdir = sys.argv[1]
rootdir = '/scratch1/Hang/Projects/deep-shape/data/modelnet40off'

classes = [s for s in os.listdir(rootdir) if os.path.isdir(os.path.join(rootdir,s))]
top_n = {'test':20,'train':80}
for (i,c) in enumerate(classes):
    for s in top_n.keys():
        curr_path = os.path.join(rootdir,c,s)
        files = os.listdir(curr_path)
        files.sort()
        files_to_rm = files[top_n[s]:]
        for f in files_to_rm:
            os.remove(os.path.join(rootdir,c,s,f))
        print('[{0}/{1}]{2}/{3}: {4} files removed.'.format(i+1,len(classes),c,s,len(files_to_rm)))

