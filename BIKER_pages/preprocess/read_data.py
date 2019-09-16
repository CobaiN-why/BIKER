import cPickle as pickle
from bs4 import BeautifulSoup
import util


def read_querys_from_file():

    path = '../data/querys_final'
    file = open(path)

    querys = list()

    while True:
        line = file.readline()
        if not line:
            break
        if len(line) <= 2 or '$$$$$' not in line:
            continue

        line = line.replace('\n','')
        line = line.split('$$$$$')

        title = line[0].strip()
        if title[0:2] == '**':
            continue
        if title[-1] == '?':
            title = title[:-1]
        apis = line[1].split(' ')
        apis_set = set()
        for api in apis:
            if len(api.strip())>1:
                apis_set.add(api)
        querys.append((title,apis_set))

    # for item in querys:
    #     print item[0],item[1]



    file.close()

    return querys

def methods_to_classes(querys):
    querys_new = list()
    for item in querys:
        #print item[0],item[1]
        methods_set = item[1]
        classes_set = set()
        for method in methods_set:
            class_name = method[:method.rfind('.')]
            classes_set.add(class_name)
        querys_new.append((item[0],classes_set))
        #print item[0],classes_set
    return querys_new

