org_path = "data/WK3l-15k/en_fr/P_en_v5.csv"
edit_path = "data/WK3l-15k/en_fr/P_en_v5_edit.csv"
original = open(org_path, "r", encoding='UTF-8')
edit = open(edit_path, "w", encoding='UTF-8')

for line in original:
    line = line.rstrip()
    triple = line.split("@@@")
    h = triple[0].replace(" ", "_")  # (_)
    r = triple[1].replace(" ", "_")
    t = triple[2].replace(" ", "_")  # \ - literals
    edit.write(h+" r/"+r+" "+t+"\n")  # relation prefix r/poles

original.close()
edit.close()
