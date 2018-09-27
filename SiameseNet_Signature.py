
# coding: utf-8

# In[1]:


from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D
from keras.models import Model, Sequential, load_model
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
import numpy.random as rng
import numpy as np
import os
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
# get_ipython().run_line_magic('matplotlib', 'inline')
def W_init(shape,name=None):
    """Initialize weights as in paper"""
    values = rng.normal(loc=0,scale=1e-2,size=shape)
    return K.variable(values,name=name)
#//TODO: figure out how to initialize layer biases in keras.
def b_init(shape,name=None):
    """Initialize bias as in paper"""
    values=rng.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,name=name)

input_shape = (105, 105, 1)
left_input = Input(input_shape)
right_input = Input(input_shape)
#build convnet to use in each siamese 'leg'
convnet = Sequential()
convnet.add(Conv2D(64,(10,10),activation='relu',input_shape=input_shape,
                   kernel_initializer=W_init,kernel_regularizer=l2(2e-4)))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(128,(7,7),activation='relu',
                   kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(128,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(256,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init))
convnet.add(Flatten())
convnet.add(Dense(4096,activation="sigmoid",kernel_regularizer=l2(1e-3),kernel_initializer=W_init,bias_initializer=b_init))

#call the convnet Sequential model on each of the input tensors so params will be shared
encoded_l = convnet(left_input)
encoded_r = convnet(right_input)
#layer to merge two encoded inputs with the l1 distance between them
L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
#call this layer on list of two input tensors.
L1_distance = L1_layer([encoded_l, encoded_r])
prediction = Dense(1,activation='sigmoid',bias_initializer=b_init)(L1_distance)
siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

optimizer = Adam(0.00006)
#//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)

siamese_net.count_params()


# ## Data 
# The data is pickled as an N_classes x n_examples x width x height array, and there is an accompanyng dictionary to specify which indexes belong to which languages.

# In[2]:


PATH = "./" #CHANGE THIS - path where the pickled data is stored


with open(os.path.join(PATH, "sig_data_train.pickle"), "rb") as f:
    (X,c) = pickle.load(f)

with open(os.path.join(PATH, "val.pickle"), "rb") as f:
    (Xval,cval) = pickle.load(f)

print("X's length {}".format(len(X)))

# print("training alphabets")  
# print("X's shape {}".format(X.shape))
# print("training alphabets")
# print(c.keys())
print("validation alphabets:")
print(cval.keys())


# In[3]:


class Siamese_Loader:
    """For loading batches and testing tasks to a siamese net"""
    def __init__(self, path, data_subsets = ["sig_data_train", "val"]):
        self.data = {}
        self.author_dict = {}
        self.info = {}
        
        for name in data_subsets:
            file_path = os.path.join(path, name + ".pickle")
            print("loading data from {}".format(file_path))
            with open(file_path,"rb") as f:
                (X,a) = pickle.load(f)
                self.data[name] = X
                self.author_dict[name] = a

    def get_batch(self,batch_size,s="sig_data_train"):
        """Create batch of n pairs, half same class, half different class"""
        X=self.data[s]
        author_dict = self.author_dict[s]
        sig_examples = len(X)
        author_examples = len(author_dict)
        w,h = X[0].shape

        #initialize 2 empty arrays for the input image batch
        pairs=[np.zeros((batch_size, h, w,1)) for i in range(2)]
        #initialize vector for the targets, and make one half of it '1's, so 2nd half of batch has same class
        targets=np.ones((batch_size,))
        targets[batch_size//2:] = 0
        
        i = 0
        #pick images from same author for the first half input
        while i < batch_size // 2:
            author_id = rng.choice(list(author_dict))
            num_of_index = len(author_dict[author_id])
            if num_of_index <= 1:
                continue
            else:                
                idx_1 = rng.randint(0, num_of_index)
                idx_2 = rng.randint(0, num_of_index)
                pairs[0][i,:,:,:] = X[author_dict[author_id][idx_1]].reshape(w, h, 1)
                pairs[1][i,:,:,:] = X[author_dict[author_id][idx_2]].reshape(w, h, 1)
                i += 1

        #pick images from differnt author for the second half input
        while i < batch_size:
            author_id_1 = rng.choice(list(author_dict))
            author_id_2 = rng.choice(list(author_dict))
            if author_id_1 == author_id_2:
                continue
            else:
                num_of_index_1 = len(author_dict[author_id_1])
                num_of_index_2 = len(author_dict[author_id_2])
                idx_1 = rng.randint(0, num_of_index_1)
                idx_2 = rng.randint(0, num_of_index_2)
                pairs[0][i,:,:,:] = X[author_dict[author_id_1][idx_1]].reshape(w, h, 1)
                pairs[1][i,:,:,:] = X[author_dict[author_id_2][idx_2]].reshape(w, h, 1)
                i += 1
        return pairs, targets
    
    def generate(self, batch_size, s="train"):
        """a generator for batches, so model.fit_generator can be used. """
        while True:
            pairs, targets = self.get_batch(batch_size,s)
            yield (pairs, targets)    

    def make_oneshot_task(self,N,s="val",language=None):
        """Create pairs of test image, support set for testing N way one-shot learning. """
        X=self.data[s]
        n_classes, n_examples, w, h = X.shape
        indices = rng.randint(0,n_examples,size=(N,))
        if language is not None:
            low, high = self.categories[s][language]
            if N > high - low:
                raise ValueError("This language ({}) has less than {} letters".format(language, N))
            categories = rng.choice(range(low,high),size=(N,),replace=False)
            
        else:#if no language specified just pick a bunch of random letters
            categories = rng.choice(range(n_classes),size=(N,),replace=False)            
        true_category = categories[0]
        ex1, ex2 = rng.choice(n_examples,replace=False,size=(2,))
        test_image = np.asarray([X[true_category,ex1,:,:]]*N).reshape(N, w, h,1)
        support_set = X[categories,indices,:,:]
        support_set[0,:,:] = X[true_category,ex2]
        support_set = support_set.reshape(N, w, h,1)
        targets = np.zeros((N,))
        targets[0] = 1
        targets, test_image, support_set = shuffle(targets, test_image, support_set)
        pairs = [test_image,support_set]

        return pairs, targets
    
    def test_oneshot_ori(self,model,N,k,s="val",verbose=0):
        """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
        n_correct = 0
        if verbose:
            print("Evaluating model on {} random {} way one-shot learning tasks ...".format(k,N))
        for i in range(k):
            inputs, targets = self.make_oneshot_task(N,s)
            probs = model.predict(inputs)
            if np.argmax(probs) == np.argmax(targets):
                n_correct+=1
        percent_correct = (100.0*n_correct / k)
        if verbose:
            print("Got an average of {}% {} way one-shot learning accuracy".format(percent_correct,N))
        return percent_correct
    
    def test_oneshot(self,model,size,s="sig_data_train"):
        print("Evaluating accuracy......")
        inputs, targets = self.get_batch(size,s)
        #inputs, targets = shuffle(inputs, targets)
        probs = model.predict(inputs)
        probs = probs > 0.5
        probs = probs.reshape(1,size)
        print("probs in test_oneshot {}".format(probs))
        print("targets in test_oneshot {}".format(targets))
        print("probs == targets in test_oneshot {}".format(probs == targets))
        corrects = np.sum(probs == targets)
        print("corrects in test_oneshot {}".format(corrects))
        
        percent_correct = (100.0 * corrects / size)
        print("Got an average of {}%  one-shot learning accuracy".format(percent_correct))
        return percent_correct
    
    def train(self, model, epochs, verbosity):
        model.fit_generator(self.generate(batch_size),
                            
                             )
    
    




def concat_images(X):
    """Concatenates a bunch of images into a big matrix for plotting purposes."""
    nc,h,w,_ = X.shape
    X = X.reshape(nc,h,w)
    n = np.ceil(np.sqrt(nc)).astype("int8")
    img = np.zeros((n*w,n*h))
    x = 0
    y = 0
    for example in range(nc):
        img[x*w:(x+1)*w,y*h:(y+1)*h] = X[example]
        y += 1
        if y >= n:
            y = 0
            x += 1
    return img


def plot_oneshot_task_original(pairs):
    """Takes a one-shot task given to a siamese net and  """
    fig,(ax1,ax2) = plt.subplots(2)
    ax1.matshow(pairs[0][0].reshape(105,105),cmap='gray')
    img = concat_images(pairs[1])
    ax1.get_yaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)
    ax2.matshow(img,cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()

def plot_batch_task(pairs):
    """Takes a one-shot task given to a siamese net and  """
    fig,(ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8) = plt.subplots(8)
    
    ax1.matshow(pairs[0][0].reshape(105,105),cmap='gray')
#     img = concat_images(pairs[1])
    ax1.get_yaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)

    ax2.matshow(pairs[1][0].reshape(105,105),cmap='gray')
#     img = concat_images(pairs[1])
    ax2.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    
    ax3.matshow(pairs[0][4].reshape(105,105),cmap='gray')
#     img = concat_images(pairs[1])
    ax3.get_yaxis().set_visible(False)
    ax3.get_xaxis().set_visible(False)

    ax4.matshow(pairs[1][4].reshape(105,105),cmap='gray')
#     img = concat_images(pairs[1])
    ax4.get_yaxis().set_visible(False)
    ax4.get_xaxis().set_visible(False)

    ax5.matshow(pairs[0][5].reshape(105,105),cmap='gray')
#     img = concat_images(pairs[1])
    ax5.get_yaxis().set_visible(False)
    ax5.get_xaxis().set_visible(False)

    ax6.matshow(pairs[1][5].reshape(105,105),cmap='gray')
#     img = concat_images(pairs[1])
    ax6.get_yaxis().set_visible(False)
    ax6.get_xaxis().set_visible(False)

    
    ax7.matshow(pairs[0][9].reshape(105,105),cmap='gray')
#     img = concat_images(pairs[1])
    ax7.get_yaxis().set_visible(False)
    ax7.get_xaxis().set_visible(False)

    ax8.matshow(pairs[1][9].reshape(105,105),cmap='gray')
#     img = concat_images(pairs[1])
    ax8.get_yaxis().set_visible(False)
    ax8.get_xaxis().set_visible(False)

    plt.xticks([])
    plt.yticks([])
    plt.show()

    





def nearest_neighbour_correct(pairs,targets):
    """returns 1 if nearest neighbour gets the correct answer for a one-shot task
        given by (pairs, targets)"""
    L2_distances = np.zeros_like(targets)
    for i in range(len(targets)):
        L2_distances[i] = np.sum(np.sqrt(pairs[0][i]**2 - pairs[1][i]**2))
    if np.argmin(L2_distances) == np.argmax(targets):
        return 1
    return 0


def test_nn_accuracy(N_ways,n_trials,loader):
    """Returns accuracy of one shot """
    print("Evaluating nearest neighbour on {} unique {} way one-shot learning tasks ...".format(n_trials,N_ways))

    n_right = 0
    
    for i in range(n_trials):
        pairs,targets = loader.make_oneshot_task(N_ways,"val")
        correct = nearest_neighbour_correct(pairs,targets)
        n_right += correct
    return 100.0 * n_right / n_trials

#Instantiate the class
loader = Siamese_Loader(PATH)
pairs,target = loader.get_batch(10)
print("No of pairs0 {}".format(len(pairs[0])))
print("No of pairs1 {}".format(len(pairs[1])))
print("target {}".format(target))
    
#example of a one-shot learning task
# pairs, targets = loader.make_oneshot_task(20,"train","Japanese_(katakana)")
# plot_oneshot_task(pairs)

pairs, targets = loader.get_batch(10)
# plot_batch_task(pairs)
inputs, targets = loader.get_batch(10,'sig_data_train')
siamese_net.load_weights('weights')
probs = siamese_net.predict(inputs)
print(probs)
import sys
sys.stdout.flush()
