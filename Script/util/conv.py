import numpy as np
def merge_layer(layers:list) -> np.array:
    """
    merge_layers to a image-like array
    :param layers: a LIST of all generated layers that want to merge.
                    With the shape(2,2) in each layer
    :return: The merged layer
    """
    newlayers = []
    ax = 0
    for item in layers:
        newlayers.append(np.expand_dims(item, axis=len(item.shape)))
        ax = len(item.shape)
    z = np.concatenate(newlayers, axis=ax)
    return z
def divide_layer(image:np.array)->list:
    """
    divide a image into layers.Each layer is given as an np.array in an array
    :param image:
    :return: a list of divided image
    """
    ans = []
    for i in range(0,image.shape[-1]):
        ans.append(image[...,i])
    return ans

def samepadding(image:np.array, core:np.array):
    cx = core.shape[0]
    cy = core.shape[1]
    ix = core.shape[0]
    iy = core.shape[1]
    lx = (cx-1)//2
    rx = cx//2
    uy = (cy-1)//2
    dy = cy//2
    if len(image.shape) == 2:
        leftmostcolumn = image[:,0]
        rightmostcolumn = image[:,-1]
        leftmostcolumn = np.expand_dims(leftmostcolumn,axis=0).T
        rightmostcolumn = np.expand_dims(rightmostcolumn,axis=0).T
        for i in range(0,lx):
            image = np.hstack((leftmostcolumn,image))
        for i in range(0,rx):
            image = np.hstack((image,rightmostcolumn))
        upmostrow = image[0,:]
        downmostrow = image[-1,:]
        for i in range(0,uy):
            image = np.vstack((upmostrow,image))
        for i in range(0,dy):
            image = np.vstack((image,downmostrow))
        return image
    else:
        layers = divide_layer(image)
        tmp = []
        for layer in layers:
           tmp.append(samepadding(layer, core))
        return merge_layer(tmp)
def conv(image:np.array,core:np.array,padding='same',scale=False)->np.array:
    """
    Now only a 2-dimentional core is supported.The shape of image can be
    a one layer grey image,or image with a lot of layers
    :param image: The image,supposed to be in shape(a,b) or (a,b,c)
    :param core:  must be 2-dimentional
    :param padding: Only same padding supported.Use "same" or None
    :return: The image after conv
    """
    imgshape = image.shape
    coreshape = core.shape
    if len(imgshape) < 2 or len(imgshape) > 4:
        raise Exception("image size wrong")
    if padding is not None and padding == 'same':
        image = samepadding(image,core)
    if len(imgshape) == 2:
        ansimg = conv22(image,core)
        if np.max(ansimg) <=255 and np.min(ansimg) >=0:
            ansimg = np.array(ansimg,dtype='u1')
    elif len(imgshape) == 3:
        ansimg = convn2(image,core)
        if np.max(ansimg) <=255 and np.min(ansimg) >=0:
            ansimg = np.array(ansimg,dtype='u1')
    else:
        ansimg = None
        print("Convolution Operation Error.")
    if scale == True:
        ansimg = touintpicture(ansimg)
    return ansimg




def convsol(image:np.array,core:np.array,start,layer=None)->int:
    if layer is not None:
        image = image[:, :, layer]
    assert len(start) == 2
    startx = start[0]
    starty = start[1]
    endx = start[0] + core.shape[1] # i : startx to endx
    endy = start[1] + core.shape[0] # j : starty to endy
    sum = 0
    for i in range(startx, endx):
        for j in range(starty, endy):
            sum = sum + image[i][j] * core[i-startx][j-starty]
    return sum

def conv22(image:np.array, core:np.array)->np.array:
    assert len(image.shape) == 2
    assert len(core.shape) == 2
    rowend = image.shape[1] - core.shape[1] # i : 0 to rowend+1
    colend = image.shape[0] - core.shape[0] # j : 0 to colend+1
    ans = []
    for i in range(0, rowend + 1):
        temp = []
        for j in range(0, colend + 1):
            temp.append(convsol(image, core, (i, j)))
        ans.append(temp)
    return np.array(ans)

def convn2(image:np.array,core:np.array)->np.array:
    layers = divide_layer(image)
    conved = []
    for layer in layers:
        conved.append(conv22(layer, core))
    return merge_layer(conved)


def medianFilter22(image:np.array,coreshape:tuple,padding=True):
    if padding:
        image = samepadding(image,np.zeros(coreshape))
    rowend = image.shape[1] - coreshape[1] # i : 0 to rowend+1
    colend = image.shape[0] - coreshape[0] # j : 0 to colend+1
    ans = []
    for i in range(0, rowend + 1):
        temp = []
        for j in range(0, colend + 1):
            temp.append(getmedian(image, coreshape, (i, j)))
        ans.append(temp)
    return np.array(ans)

def getmedian(image:np.array,coreshape:tuple,start,layer=None)->int:
    if layer is not None:
        image = image[:, :, layer]
    assert len(start) == 2
    startx = start[0]
    starty = start[1]
    endx = start[0] + coreshape[1] # i : startx to endx
    endy = start[1] + coreshape[0] # j : starty to endy
    l = []
    for i in range(startx, endx):
        for j in range(starty, endy):
            l.append(image[i][j])
    l = sorted(l)
    if len(l) % 2 != 0:
        return l[len(l) // 2]
    elif len(l) % 2 == 0:
        return (l[len(l) // 2 - 1] + l[len(l)//2]) / 2
    else:
        raise Exception

def touintpicture(sudoimg):
    minpic = np.min(sudoimg)
    maxpic = np.max(sudoimg)
    sudoimg = np.abs(sudoimg)
    if maxpic > 255:
        sudoimg = sudoimg / maxpic * 255
    img = np.array(sudoimg, dtype='u1')
    return img