#! /usr/bin/env python

import sys
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from PIL import ImageFilter
from skimage import feature, color
from sklearn.cluster import spectral_clustering
from sklearn.cluster import KMeans
from sklearn.feature_extraction import image

import matplotlib                                                                  
import matplotlib.pyplot as plt                                                    
from mpl_toolkits.mplot3d import Axes3D                                            
from matplotlib.backends.backend_pdf import PdfPages                               
import matplotlib.patches as patches                                               
from matplotlib.colors import ListedColormap                                       
import matplotlib.gridspec as gridspec                                             
import matplotlib.ticker as mtick                                                  
DEFAULT_COLORS=plt.rcParams['axes.prop_cycle'].by_key()['color']                   
from matplotlib import rc                                                          
rc('text', usetex=True) 

class ColorBooked(object):

    def __init__(self, img_fname, blur, num_centroids):

        # image read with PIL
        self.img_fname = img_fname
        self.im = Image.open(img_fname)
        self.ypix, self.xpix = self.im.size
        self._preblur_img(blur) 

        # store raw image as numpy array
        self.raw_img = np.asarray(self.im) 
        print(np.shape(self.raw_img))
        self.num_centroids = num_centroids
        self._normalize_img()

        print("Raw image <%s> read in:"%(self.img_fname))
        print("Pixels (width=%d x height=%d):"%(self.xpix, self.ypix))

        
    def _preblur_img(self, blur):
        """
        Apply some Gaussian blur to image to smooth out tediously small features
        that detract from quality of final black and white outline
        
        Hard to automate this parameter, several values should be tested
        to obtain the best coloring book image 
        """
        if(blur==None):
            # heuristic blur if none provided
            blur=max(self.xpix,self.ypix)/500

        self.blur = blur
        self.im = self.im.filter(ImageFilter.GaussianBlur(blur))


    def _normalize_img(self):
        # make sure RGB color scale is (0,1)
        # generate flattened img representation for processing
        self.max_depth = np.max(np.max(np.max(self.raw_img)))
        self.scaled_img = self.raw_img/self.max_depth
        self.flattened = np.reshape(self.scaled_img, (self.xpix*self.ypix,3))


    ###########################################################################
    # K-MEANS related functions

    def _init_kmeans_random(self):
        self.init_centroids = np.array([[np.random.random(),
                                         np.random.random(),
                                         np.random.random()]
                                        for k in range(self.num_centroids)], 
                                         dtype='float32')

    def _init_kmeans_preconditioned(self):
        # important to have as diverse as possible initial guesses for centroids
        # if not, k-means may miss an important color if it only constitutes
        # a small portion of the image

        self.init_centroids = np.array([[k/self.num_centroids,
                                         k/self.num_centroids,
                                         k/self.num_centroids]
                                        for k in range(self.num_centroids)], 
                                         dtype = 'float32')

    def input_fn(self):
        return tf.train.limit_epochs(
            tf.convert_to_tensor(self.flattened, dtype=tf.float32), num_epochs=1)

    
    def _train_kmeans_tf(self, max_iterations=10):
        self.max_iterations = max_iterations 

        self.kmeans = tf.contrib.factorization.KMeansClustering(
            num_clusters=self.num_centroids, 
            initial_clusters = "kmeans_plus_plus",
            use_mini_batch=False)

        previous_centers = None
        for _ in range(self.max_iterations):
            self.kmeans.train(self.input_fn)
            cluster_centers = self.kmeans.cluster_centers()
            if previous_centers is not None:
                print('delta: %s'%(cluster_centers - previous_centers))
            previous_centers = cluster_centers
            print('score: %s'%(self.kmeans.score(self.input_fn)))
        self.cluster_centers = cluster_centers
        print('cluster centers: %s'%(cluster_centers))
        self.cluster_indices = list(self.kmeans.predict_cluster_index(self.input_fn))

    def _train_kmeans_sk(self, max_iterations =10):
        self.kmeans = KMeans(n_clusters=self.num_centroids, init='k-means++', 
                             max_iter=max_iterations, n_init=3)
        self.cluster_indices=self.kmeans.fit_predict(self.flattened)
        self.cluster_centers=self.kmeans.cluster_centers_


    ###########################################################################




    ###########################################################################
    # Spectral clusteringd related functions

    def _train_spectral_clustering(self):

        graph = image.img_to_graph(self.flattened)
        labels = spectral_clustering(graph, n_clusters=self.num_centroids, 
                                     eigen_solver='arpack')
        print(labels)

    ###########################################################################


    ###########################################################################
    # GMM 
    # https://github.com/aakhundov/tf-example-models/blob/master/models/tf_kmeans.py

    def _train_gmm(self):
        DIMENSIONS = 3
        CLUSTERS = self.num_centroids
        DATA_POINTS = len(self.flattened)
        data = self.flattened

        TRAINING_STEPS = 20
        TOLERANCE = 0

        input = tf.placeholder(tf.float32, [None, DIMENSIONS])

        # trainable variables: clusters means
        random_point_ids = tf.squeeze(tf.multinomial(\
                            tf.ones([1, tf.shape(input)[0]]), CLUSTERS)\
                           )
        means = tf.Variable(tf.gather(input, random_point_ids), dtype=tf.float32)

        # E-step: recomputing cluster assignments according to the current means
        inputs_ex, means_ex = tf.expand_dims(input, 0), tf.expand_dims(means, 1)
        distances = tf.reduce_sum(tf.squared_difference(inputs_ex, means_ex), 2)
        assignments = tf.argmin(distances, 0)

        # M-step: relocating cluster means according to the computed assignments
        sums = tf.unsorted_segment_sum(input, assignments, CLUSTERS)
        counts = tf.reduce_sum(tf.one_hot(assignments, CLUSTERS), 0)
        means_ = tf.divide(sums, tf.expand_dims(counts, 1))

        # distortion measure: sum of squared distances 
        # from each point to the closest cluster mean
        distortion = tf.reduce_sum(tf.reduce_min(distances, 0))

        # updating the means by new values
        train_step = means.assign(means_)


        # RUNNING COMPUTATIONAL GRAPH

        # creating session
        sess = tf.InteractiveSession()

        # initializing trainable variables
        sess.run(tf.global_variables_initializer(), feed_dict={input: data})

        previous_assignments = None

        # training loop
        for step in range(TRAINING_STEPS):
            # executing a training step and
            # fetching evaluation information
            distortion_measure, current_means, current_assignments, _ = sess.run(
                [distortion, means_, assignments, train_step],
                feed_dict={input: data}
            )

            if step > 0:
                # computing the number of re-assignments during the step
                re_assignments = (current_assignments != previous_assignments).sum()
                print("{0}:\tdistortion {1:.2f}\tre-assignments {2}".format(
                    step, distortion_measure, re_assignments))

                # stopping if no re-assignments occurred
                if re_assignments <= TOLERANCE:
                    break
            else:
                print("{0}:\tdistortion {1:.2f}".format(
                    step, distortion_measure))

            previous_assignments = current_assignments

        self.cluster_centers = current_means
        self.cluster_indices = current_assignments



    ###########################################################################
    # post-algorithm classification

    def _map_pixels_to_centroids(self):

        self.colorized = np.zeros(np.shape(self.flattened),dtype='uint8')
        for i, pixel in enumerate(self.flattened):
            cluster_index = self.cluster_indices[i]
            center = np.array(self.cluster_centers[cluster_index]*255, 
                              dtype='uint8')
            self.colorized[i] = center

    ###########################################################################


    ###########################################################################
    # run algorithms 

    def run_kmeans_training(self, engine='tf'):
        #self._init_kmeans_random()
        self._init_kmeans_preconditioned()
        if(engine=='tf'):
            self._train_kmeans_tf()
        elif(engine=='sk'):
            self._train_kmeans_sk()
        self._map_pixels_to_centroids()
        self.img_final = np.reshape(self.colorized,(self.xpix, self.ypix, 3))
        print("Final image:")
        print(self.img_final)

    def run_gmm(self):
        self._train_gmm()
        self._map_pixels_to_centroids()
        self.img_final = np.reshape(self.colorized,(self.xpix, self.ypix, 3))
        print("Final image:")
        print(self.img_final)

    def run_spectral_clustering(self):
        self._train_spectral_clustering()

    ###########################################################################


    ###########################################################################
    # image post-processing 

    def write_image(self, outname = None, ext = 'png'):

        last_dot = self.img_fname.rfind('.')
        if(last_dot==-1):
            last_dot = len(self.img_fname)

        if(outname is None):
            # autogen output filename if none given by user 
            outname = self.img_fname[:last_dot]+'_colorized%d.%s'\
                      %(self.num_centroids,ext)

        final_colors = set()
        for i in range(np.shape(self.img_final)[0]):
            for j in range(np.shape(self.img_final)[1]):
                color = tuple(self.img_final[i,j])
                final_colors.add(color)

        print("Writing color-segmented image to: %s"%(outname))
        print("Final color list: %s"%(final_colors))
        print("Number of colors: %s"%(len(final_colors)))

        im = Image.fromarray(self.img_final)
        im.save(outname) 

        #im = Image.open(outname)
        #double_check = np.asarray(im)
        #double_check_set = set()
        #for i in range(np.shape(double_check)[0]):
        #    for j in range(np.shape(double_check)[1]):
        #        color = tuple(double_check[i][j])
        #        double_check_set.add(color)

        #print("Double checking color-segmented image at: %s"%(outname))
        #print("Final color list: %s"%(double_check))
        #print("Number of colors: %s"%(len(double_check_set)))

        #if(len(double_check_set) != len(final_colors)):
        #    print("WARNING: pixel values changing during write/read process")


    def write_outline_rgb(self, sigma=0.1, outname=None, ext='jpg'):

        last_dot = self.img_fname.rfind('.')
        if(last_dot==-1):
            last_dot = len(self.img_fname)

        if(outname is None):
            # autogen output filename if none given by user 
            outname1 = self.img_fname[:last_dot]+\
                       '_outlined%d_blur%1.1f_canny%1.1f.%s'\
                       %(self.num_centroids,self.blur,sigma,ext)

        edges_r = feature.canny(self.img_final[:,:,0],sigma=sigma)
        edges_g = feature.canny(self.img_final[:,:,1],sigma=sigma)
        edges_b = feature.canny(self.img_final[:,:,2],sigma=sigma)

        final_edges = np.logical_or(edges_r, edges_g)
        final_edges = np.logical_or(final_edges, edges_b) 
        final_edges = np.array(np.logical_not(final_edges)*self.max_depth,
                               dtype = 'uint8')

        final_edges[:,0] = 0
        final_edges[:,np.shape(final_edges)[1]-1] = 0
        final_edges[0,:] = 0
        final_edges[np.shape(final_edges)[0]-1,:] = 0

        towrite = Image.fromarray(final_edges)
        towrite.save(outname1)


    def write_color_pallete(self, outname = None, ext = 'jpg'):
        """
        Write image that just has blocks for each color in the final image
        """

        last_dot = self.img_fname.rfind('.')
        if(last_dot==-1):
            last_dot = len(self.img_fname)

        if(outname is None):
            # autogen output filename if none provided
            outname = self.img_fname[:last_dot]+'_pallete%d.%s'%(self.num_centroids,ext)

        curr_fig=0                                                              
        plt.figure(curr_fig,figsize=(7,0.6))                                    
        gs = gridspec.GridSpec(1,self.num_centroids)
        gs.update(left=0.09, right=0.95, top=0.95, bottom=0.15,wspace=0.3)      
        for i in range(self.num_centroids):
            ax = plt.subplot(gs[i])
            xlim=ax.get_xlim()
            ylim=ax.get_ylim()
            ax.fill_between(xlim,[ylim[0],ylim[0]],[ylim[1],ylim[1]],color=self.cluster_centers[i])

            ax.xaxis.set_major_formatter(plt.NullFormatter())
            ax.yaxis.set_major_formatter(plt.NullFormatter())

        plt.savefig(outname)
        plt.close()
            

    def visualize_results_3D(self):
        """
        Visualize the RGB maps of the original image and the color book image
        """

        curr_fig=1
        fig = plt.figure(curr_fig)
        ax1=fig.add_subplot(111,projection='3d')

        # random subset of pixels with maximum number specified
        max_pts=10000
        if(len(self.flattened)<max_pts):
            max_pts=len(self.flattened)
        inds=np.random.permutation([i for i in range(len(self.flattened))])[0:max_pts]
        reduced=self.flattened[inds]
        reducedc=self.colorized[inds]

        for i in range(len(reduced)):
            if(reduced[i,1]>1.5*reduced[i,0] and reduced[i,1]>1.5*reduced[i,2]):
                reducedc[i]=reduced[i]*255
        
        ax1.scatter(reduced[:,0],reduced[:,1],reduced[:,2],
                    c=[(reducedc[i,0]/255,reducedc[i,1]/255,reducedc[i,2]/255) for i in range(len(reducedc))])

        ax1.set_xlabel("R (x)")
        ax1.set_ylabel("G (y)")
        ax1.set_zlabel("B (z)")

        curr_fig+=1
        fig = plt.figure(curr_fig)
        ax1=fig.add_subplot(111,projection='3d')
        
        ax1.scatter(reduced[:,0],reduced[:,1],reduced[:,2],
                    c=[(reduced[i,0],reduced[i,1],reduced[i,2]) for i in range(len(reduced))])

        ax1.set_xlabel("R (x)")
        ax1.set_ylabel("G (y)")
        ax1.set_zlabel("B (z)")

        plt.show()

    ###########################################################################


    def generate_colorbook_image(self):
        
        self.run_kmeans_training(engine='sk')
        #self.run_gmm()
        self.write_image()
        self.write_outline_rgb()
        self.write_color_pallete()
        self.visualize_results_3D()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='image to process')
    parser.add_argument('--blur', type=float, default=None,
                        help='StDev of Gaussian blur to preprocess image')
    parser.add_argument('--colors', type=int, default=10,
                        help='Number of colors in coloring book image')
    args = parser.parse_args()

    colorbook = ColorBooked(args.image, args.blur, args.colors)
    colorbook.generate_colorbook_image()



        
