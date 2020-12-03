from unittest import TestCase
import numpy as np
from Script.lab2.histogram import leveled
from Script.lab2.histogram import gettable
from Script.lab2.histogram import restored
from Script.lab2.histogram import histogram_equalization
class Test(TestCase):
    def test_leveled(self):
        img = np.array([[0, 255, 255, 0], [63, 64, 65, 66], [192, 193, 194, 224], [0, 0, 0, 0]])
        ans = np.array([[0,7,7,0],[1,2,2,2],[6,6,6,7],[0,0,0,0]])
        assert (leveled(img,gettable(8)) == ans).all()

    def test_restored(self):
        ans = np.array([[0, 7, 7, 0], [1, 2, 2, 2], [6, 6, 6, 7], [0, 0, 0, 0]])
        img = np.array([[0,224,224,0],[32,64,64,64],[192,192,192,224],[0,0,0,0]])
        assert (restored(ans,gettable(8)) == img).all()

    def test_histogram_equalization(self):
        img = np.array([[0, 0, 0, 0], [0, 32, 32, 32], [32, 32, 32, 32], [32, 64, 32, 254]])
        img = histogram_equalization(img,8)
        print(img)