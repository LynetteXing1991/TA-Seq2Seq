import os
import time
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from progressbar import ProgressBar
from PIL import Image
import sys
import configurations
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--proto",  default="normal_adagrad",
    help="Prototype config to use for config")
args = parser.parse_args()


def showmat(name, mat, alpha, beta):
    sample_dir = 'samples/'
    alpha = [a.decode('utf-8') for a in alpha]
    beta = [b.decode('utf-8') for b in beta]

    fig = plt.figure(figsize=(20, 20), dpi=80)
    plt.clf()
    matplotlib.rcParams.update({'font.size': 18})
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    ax.xaxis.tick_top()
    res = ax.imshow(mat, cmap=plt.cm.Blues, 
                    interpolation='nearest')
    
    font_prop = FontProperties()
    font_prop.set_file('./wqy-zenhei.ttf')
    font_prop.set_size('large')
    
    plt.xticks(range(len(alpha)), alpha, rotation=60, 
               fontproperties=font_prop)
    plt.yticks(range(len(beta)), beta, fontproperties=font_prop)

    cax = plt.axes([0.0, 0.0, 0.0, 0.0])
    plt.colorbar(mappable=res, cax=cax)
    plt.savefig(name + '.png', format='png')
    plt.close()


def main(config):
    images_dir = config['attention_images']
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)

    source_file = open(config['val_set_source'], 'r').readlines()
    target_file = pickle.load(open(config['val_output_repl'] + '.pkl', 'rb'))
    weights = pickle.load(open(config['attention_weights'], 'rb'))

    pbar = ProgressBar(max_value=len(source_file)).start()
    for i, (source, target, weight) in enumerate(
            zip( source_file, target_file, weights)):
        pbar.update(i + 1)
        source = source.strip().split()
        showmat(images_dir + str(i), weight, source, target)
    pbar.finish()

def crop(config):
    indir = config['attention_images']
    outdir = config['attention_images'] + '/cropped'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for fname in os.listdir(indir):
        inpath = os.path.join(indir, fname)
        outpath = os.path.join(outdir, fname)
        if os.path.isdir(inpath):
            continue
        image = Image.open(inpath)
        w, h = image.size
        image = image.crop((0, 0, w, h-12))
        image.save(outpath, 'png')


if __name__ == '__main__':
    config = getattr(configurations, args.proto)()
    main(config)
    crop(config)
