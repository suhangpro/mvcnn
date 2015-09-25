#!/usr/bin/python

import os
import os.path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataDir')
parser.add_argument('-o','--output')
parser.add_argument('-n','--nlargest',type=int,default=100)
args = parser.parse_args()

allClasses = filter(lambda s: os.path.isdir(os.path.join(args.dataDir,s)),os.listdir(args.dataDir))
allClassesCnt = {x: len(os.listdir(os.path.join(args.dataDir,x))) for x in allClasses}
resultClasses = sorted(allClassesCnt,key=allClassesCnt.__getitem__,reverse=True)[:args.nlargest]

if args.output==None: 
    for x in resultClasses: 
        print x
else: 
    f = open(args.output,'w')
    f.write("\n".join(resultClasses)+"\n")
    f.close()
