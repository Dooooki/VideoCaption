import json


def convert_txt_to_json(data_root, txtName):
    '''
    convert caption.txt to caption.json
    Args:
        dataRoot(str): path of caption.txt
        txtName(str): name of caption.json
    '''
    f = open(data_root+txtName, 'r')
    temp = f.readlines()
    data = list(map(lambda x: x.strip().split(' ', 1), temp))

    capDict = {}

    caps = []
    vid = data[0][0]
    caps.append(data[0][1])
    for item in data[1:]:
        cur_vid, cap = item
        if cur_vid != vid:
            capDict[vid] = caps
            caps = []
            vid = cur_vid
        caps.append(cap)
    capDict[vid] = caps

    json_str = json.dumps(capDict)
    with open(data_root+'caption.json', 'w') as j:
        j.write(json_str)
