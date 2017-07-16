import caffe
import numpy as np

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def compute_hist(bottom):
    n_cl = bottom[0].data.shape[1]
    hist = np.zeros((n_cl, n_cl))
    for i in range(bottom[0].data.shape[0]):
        
        hist += fast_hist(bottom[1].data[0].flatten(),
                                bottom[0].data[0].argmax(0).flatten(),
                                n_cl)
    return hist


def seg_tests(bottom):
    n_cl = bottom[0].data.shape[1]
    hist = compute_hist(bottom)
    with np.errstate(divide='ignore',invalid='ignore'):
        iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        mean_iu = np.nanmean(iu)

    return mean_iu


class Iou(caffe.Layer):
        
    def setup(self,bottom,top):
        pass
    
    def reshape(self,bottom,top):
        for i in range(len(top)):
            top[i].reshape((1))        
    
    def forward(self,bottom,top):
        mean_iu = seg_tests(bottom)

        top[0].data[0] = mean_iu

    def backward(self,top,propagate_down,bottom):
        pass

