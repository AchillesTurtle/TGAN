import pickle
import keras.datasets.cifar10
def _unpickle(filename):
    """
    Unpickle the given file and return the data.
    Note that the appropriate dir-name is prepended the filename.
    """

    # Create full path for the file.

    print("Loading data: " + filename)

    with open(filename, mode='rb') as file:
        # In Python 3.X it is important to set the encoding,
        # otherwise an exception is raised here.
        print(file)
        data = pickle.load(file, encoding='bytes')

    return data

def load_cifar10():
    train,test=keras.datasets.cifar10.load_data()
    train_data,train_label=train[0],train[1]
    test_data,test_label=test[0],test[1]
    labeled_data,labeled_label=train_data[0:1000,:,:,:],train_label[0:1000]
    unlabeled_data=train_data[1000:,:,:,:]
    print(labeled_data.shape)
    return labeled_data,labeled_label,unlabeled_data,test_data,test_label
if __name__=='__main__':
    load_cifar10()