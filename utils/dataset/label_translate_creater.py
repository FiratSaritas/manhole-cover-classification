import pickle


label_translate_1 = {
    'Rost/Strassenrost': 0,
    'Vollguss/Pickelloch belueftet': 1,
    'Gussbeton/Pickelloch geschlossen': 2,
    'Vollguss/Pickelloch geschlossen': 3,
    'Gussbeton/Pickelloch belueftet': 4,
    'Vollguss/Handgriff geschlossen': 5,
    'Gussbeton/Handgriff seitlich': 6,
    'Andere/-': 7,
    'Rost/Einlauf rund': 8,
    'Rost/Strassenrost gewoelbt': 9,
    'Vollguss/Aufklappbar': 10,
    'Gussbeton/Handgriff mitte': 11,
    'Vollguss/Handgriff geschlossen, verschraubt': 12
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
    label_save(label_translate_1,'label_translate.pkl')

