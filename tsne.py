# create a t-sne plot 

import torch 
import numpy as np
import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

colors_per_class = {
    0 : [254, 202, 87],
    1 : [255, 107, 107],
    2 : [10, 189, 227],
    3 : [255, 159, 243],
    4 : [16, 172, 132],
    5 : [128, 80, 128],
    6 : [87, 101, 116],
    7 : [52, 31, 151],
    8 : [0, 0, 0],
    9 : [100, 100, 255],
}

def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

def visualize_tsne_points(tx, ty, labels,save_path):
    # initialize matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:
        # find the samples of the current class in the data
        # print("label",label,len(labels))
        indices = [i for i, l in enumerate(labels) if l == label]
        # print(indices)
        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format:
        # BGR -> RGB, divide by 255, convert to np.array
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255

        # add a scatter plot with the correponding color and label
        ax.scatter(current_tx, current_ty, c=color, label=label)

    # build a legend using the labels we set previously
    ax.legend(loc='best')

    # finally, show the plot
    # plt.show()
    fig.savefig(save_path)

    
def draw_tsne(model,dataloader,embedding_label,save_path='visualization/tsne.png'):
    model.eval()
    features = None
    labels= []
    for batch_idx,batch in enumerate(dataloader):
        print("rajat",batch_idx)
        images, targets, meta= batch
        images=images.to(device) 
        images= images.permute(0, 3, 1, 2)
        labels+=targets.tolist()
        n_repeats= images.shape[0]
        embedding_labels = torch.ones(1)*embedding_label
        embedding_labels= embedding_labels.type(torch.LongTensor).to(device)
        embedding_labels = torch.cat(n_repeats*[embedding_labels])
        embedding_labels = embedding_labels.unsqueeze(1)
        with torch.no_grad():
            output = model(images,embedding_labels)
        current_features = output.cpu().numpy()
        if features is not None:
            features = np.concatenate((features, current_features))
        else:
            features = current_features

    #TSNE fit transform 
    print("calling tsne fit transfrom ",features.shape)
    #sklearn
    #from sklearn.manifold import TSNE
    #tsne = TSNE(n_components=2).fit_transform(features)
    
    #tsnecuda
    from tsnecuda import TSNE
    tsne = TSNE(n_components=2).fit_transform(features)
    print("tsne shape",tsne.shape)
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    
    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    print("tx",tx)
    print("labels",len(labels),labels)
    visualize_tsne_points(tx, ty,labels,save_path)