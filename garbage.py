    node = InfoNode(None, [], '0'*(len(listleaves[0][0])-1))
    prob = []
    for i in range(2):
        name = node.name+str(i)
        for j in range(len(listleaves)):
            if name == listleaves[j][0]:
                prob.append(listleaves[j][1])
    if abs(prob[0]-prob[1]) > epsilon:
        print "Error", node.name, prob[0], prob[1]
    else:
        prob[1]=1-prob[0]

    for i in range(2):
        name = node.name+str(i)
        if name in listnodesname:
            mynode = listnodes[listnodesname.index(name)]
            node.childs.append([mynode, prob[i]]) 
        else:
            mynode=InfoNode(None, [], name, 0)
            listnodesname.append(name)
            listnodes.append(mynode)
            node.childs.append([mynode, prob[i]]) 
            treetomarkov(mynode, listleaves, listnodesname, listnodes) 
    return node


