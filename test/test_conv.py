from unittest import TestCase
from Script.util.conv import convsol
from Script.util.conv import conv22
from Script.util.conv import merge_layer
from Script.util.conv import divide_layer
from Script.util.conv import convn2
from Script.util.conv import samepadding
from Script.util.conv import getmedian
from Script.util.conv import medianFilter22
import numpy as np
from Script.util.conv import conv
class Test(TestCase):
    def test_merge_layer(self):
        x = np.array([[1, 1, 1], [1, 1, 1],[1, 1, 1]])
        y = np.array([[1, 0, 0], [0, 1, 0],[0, 0, 1]])
        z = merge_layer([x,y])
        assert (z == np.array([[[1,1],[1,0],[1,0]],[[1,0],[1,1],[1,0]],[[1,0],[1,0],[1,1]]])).all()
    def test_divide_layer(self):
        z = np.array([[[1,1],[1,0],[1,0]],[[1,0],[1,1],[1,0]],[[1,0],[1,0],[1,1]]])
        x = divide_layer(z)
        assert (x[0] == np.array([[1, 1, 1], [1, 1, 1],[1, 1, 1]])).all()
        assert (x[1] == np.array([[1, 0, 0], [0, 1, 0],[0, 0, 1]])).all()
    def test_convsol(self):
        image = np.array([[1, 0, 0, 1], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
        core = np.array([[1,0,0],[0,1,0],[0,0,1]])
        # print(convsol(image,core,(0,0)))
        assert convsol(image, core, (0, 0)) == 3
        assert convsol(image, core, (0, 1)) == 0
        assert convsol(image, core, (1, 0)) == 3
        assert convsol(image, core, (1, 1)) == 3
        image = np.array([[[1,1], [1,0], [1,0], [1,1]], [[1,1], [1,1], [1,0], [1,0]], [[1,0], [1,1], [1,1], [1,0]], [[1,0], [1,0], [1,1], [1,1]]])
        core = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert convsol(image, core, (0, 0), 1) == 3
        assert convsol(image, core, (0, 1), 1) == 0
        assert convsol(image, core, (1, 0), 1) == 3
        assert convsol(image, core, (1, 1), 1) == 3

    def testconv22(self):
        image = np.array([[1, 0, 0, 1], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
        core = np.array([[1,0,0],[0,1,0],[0,0,1]])
        result = conv22(image,core)
        assert (result == np.array([[3,0],[3,3]])).all()

    def testconvn2(self):
        image = np.array([[[1,1], [1,0], [1,0], [1,1]], [[1,1], [1,1], [1,0], [1,0]], [[1,0], [1,1], [1,1], [1,0]], [[1,0], [1,0], [1,1], [1,1]]])
        core = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        ans = convn2(image,core)
        ans = divide_layer(ans)
        assert (ans[0] == np.array([[3, 3], [3, 3]])).all()
        assert (ans[1] == np.array([[3, 0], [3, 3]])).all()
    def testpadding(self):
        image = np.array([[1, 0, 0, 1], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
        core = np.array([[1,1,1],[1,1,1],[1,1,1]])
        newimage = samepadding(image, core)
        assert (newimage == np.array([[1,1,0,0,1,1],[1,1,0,0,1,1],[1,1,1,0,0,0],[0,0,1,1,0,0],[0,0,0,1,1,1],[0,0,0,1,1,1]])).all()
        image = np.array(
            [[[1, 1], [1, 0], [1, 0], [1, 1]], [[1, 1], [1, 1], [1, 0], [1, 0]], [[1, 0], [1, 1], [1, 1], [1, 0]],
             [[1, 0], [1, 0], [1, 1], [1, 1]]])
        core = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        newimage = samepadding(image, core)
        layer1 = np.array([[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1]])
        layer2 = np.array([[1,1,0,0,1,1],[1,1,0,0,1,1],[1,1,1,0,0,0],[0,0,1,1,0,0],[0,0,0,1,1,1],[0,0,0,1,1,1]])
        assert (newimage == merge_layer([layer1, layer2])).all()
    def testconvtest(self):
        image = np.array([[1,3,7,6],[5,4,2,8],[0,3,9,7],[4,6,2,4]])
        core = np.array([[0,1,0],[1,3,1],[2,4,5]])
        ans = np.array([[79,102],[64,79]])
        assert (conv(image,core,padding=None) == ans).all()

    def testgetmedian(self):
        image = np.array([[1, 3, 7, 6], [5, 4, 2, 8], [0, 3, 9, 7], [4, 6, 2, 4]])
        median = getmedian(image,(3,3),(0,0))
        assert median == 3
    def testmedianFilter22(self):
        image = np.array([[1, 3, 7, 6], [5, 4, 2, 8], [0, 3, 9, 7], [4, 6, 2, 4]])
        ans = medianFilter22(image, (3, 3),padding=False)
        assert (ans == np.array([[3,6],[4,4]])).all()
