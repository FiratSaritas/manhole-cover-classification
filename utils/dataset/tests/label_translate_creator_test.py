import pickle
import unittest
import sys
sys.path.append('../')
from label_translate_creater import label_save


class TestLabelFile(unittest.TestCase):
    def test_label_save(self):
        pkl_url_file = "label_translate_test.pkl"
        label_translate_1 = {
            'Rost/Strassenrost': 0
        }

        label_save(label_translate_1,pkl_url_file)

        with open(pkl_url_file, 'rb') as pkl_file:
            label_dict = pickle.load(pkl_file)

        self.assertEqual(type(label_dict), dict)

