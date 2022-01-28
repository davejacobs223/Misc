#! bin/usr/env python 


import numpy as np
import csv
from glob import glob





# CO=input('Trial Start State:')
# CP=input('Pretrial State:')
# AE=input('Action State:')
# FD=input('Food Drop State:')
# FP=input('Pre Pellet Dropn State:')
# FRe=input('Food Retrieval State:')

####
def getlatencies(dataset,startrt,prior,endrt,filetype):
    actionlist=[]
    pells=0
    mark=0
    IApokes=0
    actionmean='Nan'
    for sample in range(len(dataset)):
            if dataset[sample][9] == startrt and data2[sample][8] in prior:
                    mark=1
                    if filetype == 'txt':
                        start=float(data2[sample][7])*20*.001
                    elif filetype == 'csv':
                        start=float(data2[sample][7])
            elif dataset[sample][9] == endrt and mark == 1:
                    if filetype == 'txt':
                        end=float(data2[sample][7])*20*.001
                        latency=end-start
                        actionlist.append(latency)
                        actionmean=np.mean(actionlist)
                        pells=pells+1
                        mark=0
                    elif filetype == 'csv':
                        end=float(data2[sample][7])
                        latency=end-start
                        actionlist.append(latency)
                        actionmean=np.mean(actionlist)
                        pells=pells+1
                        mark=0           
            elif dataset[sample][8] == startrt and dataset[sample][12] == '1':
                IApokes=IApokes+1
    return (actionlist,actionmean,np.std(actionlist),pells,IApokes)
###

masterlist=[]
a=glob('DailyData2/*.txt')
a=a+glob('DailyData2/*.csv')#add *row
for sess in a:
    ftype=sess.split('.')[1]
    data=[]
    if ftype == 'txt':
        with open(sess,'r') as csvfile:
        	reader=csv.reader(csvfile,delimiter='\t')
        	for line in reader:
        		data.append(line)
        	csvfile.close()
    elif ftype == 'csv':
        with open(sess,'r') as csvfile:
            reader=csv.reader(csvfile)
            for line in reader:
                data.append(line)
            csvfile.close()        
    data2 =np.array(data)
    session=sess.split('\\')[1].split('.')[0].split('_')[1][1:]
    subject=sess.split('\\')[1].split('.')[0].split('_')[0]+sess.split('\\')[1].split('.')[0].split('_')[1]    
    snumber=sess.split('_')[2].split('.')[0]
    
    
    if subject == 'S162':
        ActionRT=getlatencies(data2,'3',['6','3'],'4',ftype)
        FoodRT=getlatencies(data2,'4',['3'],'6',ftype)
    else:
        ActionRT=getlatencies(data2,'2',['1','4'],'3',ftype)
        FoodRT=getlatencies(data2,'3',['2'],'4',ftype)
    Actionmean=ActionRT[1]
    Actionstd=ActionRT[2]
    Foodmean=FoodRT[1]
    Foodstd=FoodRT[2]
    Pellets=ActionRT[3]
    InactivePokes=ActionRT[4]
    masterlist.append([subject,snumber,Actionmean,Actionstd,Foodmean,Foodstd,Pellets,InactivePokes])


with open('FR1Metrics.csv', 'w') as file:
    wr=csv.writer(file,lineterminator='\n')
    wr.writerow(['Subject','Session','Action_Mean','Action_std','Food_Mean','Food_Std','Pellets','InactivePokes'])
    for item in masterlist:
        wr.writerow(item)
