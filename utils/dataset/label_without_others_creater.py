import pickle
import sys
sys.path.append('../')


label_translate_1 = {
    'Rost/Strassenrost': 0,
    'Vollguss/Pickelloch belueftet': 1,
    'Gussbeton/Pickelloch geschlossen': 2,
    'Vollguss/Pickelloch geschlossen': 3,
    'Gussbeton/Pickelloch belueftet': 4,
    'Vollguss/Handgriff geschlossen': 5,
    'Gussbeton/Handgriff seitlich': 6,
    'Rost/Einlauf rund': 7,
    'Rost/Strassenrost gewoelbt': 8,
    'Vollguss/Aufklappbar': 9,
    'Gussbeton/Handgriff mitte': 10,
    'Vollguss/Handgriff geschlossen, verschraubt': 11
}

def label_save(label_translate, name):
    """save label_translate as pickle file

    Parameters
    ----------
    label_translate : dictionary
        dictionary with key as label and value as index
    name : string
        name of the pickle file
    """
    with open(name, 'wb') as f:
        pickle.dump(label_translate, f)

if __name__ == '__main__':
    label_save(label_translate_1,'label_without_others.pkl')
    