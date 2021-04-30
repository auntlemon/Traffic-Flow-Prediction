import numpy as np
from geopy.distance import geodesic

w_dis = np.eye(202)
jw = []

# i = 0
with open('data/hz/road_geo.txt') as f:
    for line in f:
        a = line.strip().split(',')

        jw.append(a)
        # i = i+1
        # print(a)
#print(jw)

sum = 0
for p in range(202):
    # print('第'+str(p)+'个')

    for q in range(len(jw[p])-2):
        e = jw[p][q+1].split(' ')
        d = jw[p][q+1+1].split(' ')
        # print(e)
        # print(d)
        dis = geodesic((e[1], e[0]), (d[1], d[0])).km
        # print(dis)
        sum = sum + dis
#print(sum/202)

mean_dis = round(sum/202,3)
# print(np.array(jw).shape)
for p in range(202):
    for q in range(p+1,202):
        #print('第' + str(p) + '个和第' + str(q)+ '个')
        min = 99999999
        for i in range(len(jw[p])-1):
            for j in range(len(jw[q])-1):
                c = jw[p][i+1].split(' ')
                d = jw[q][j+1].split(' ')
                # print(c)
                # print(d)
                dis = round(geodesic((c[1], c[0]), (d[1], d[0])).km,3)
                # print(dis)
                if min > dis:
                    min = dis
        # print(min)
        w_dis[p][q] = round(1/(1+min/mean_dis),5)
        w_dis[q][p] = round(1 / (1 + min / mean_dis), 5)
        #print(w_dis[p][q])
#print(w_dis)
res=True
for i in range(202):
    for j in range(202):
        if w_dis[i][j] != w_dis[j][i]:
            res = False
            break
#print(res)

np.savetxt('data/adj_distance.txt',w_dis,fmt='%.5f')
