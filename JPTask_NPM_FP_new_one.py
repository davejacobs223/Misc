#! bin/usr/env python 

#A Fiber Photometry Script that is a work in progress

import scipy
import csv
import numpy as np
from numpy import mean
from pylab import *
from scipy import signal
from statsmodels import robust

# from TDTbin2py import TDTbin2py #make sure these are in the MyNotebook directory
# import pprint
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
# 
#  for a in offset:
#     plt.plot(a,.15,'-g|',mew=4,ms=10))

######################################################################################33




#get timestamp data and load files
a='corrNpokeCue2019-03-05T09_59_55.csv'#cue file for GT
b='feedercue2019-03-05T09_59_55.csv'
d='HMSR251_JPT_Day12019-03-05T09_59_55.csv'#data file of camera
CueStamps='251_JP1_Cues.csv'
SeekStamps='251_JP1_Seeks.csv'
FoodDrop='251_JP1_Food.csv'
FoodRet='251_JP1_Retreivals.csv'
#MBFile='14_sts8_MB.csv'#Microbehavior file

data=[]

with open(a,'r') as csvfile:
    reader=csv.reader(csvfile,delimiter=' ')
    for line in reader:
        data.append(line)
    csvfile.close()

data2 =np.array(data)


#get first cue as Ground Truth
StartCue=(float(data2[1][0])/1000)-(float(data2[0][0])/1000)+1.4
offset=StartCue
###
#find the trial specific ground truth # if there is a dropped trial we will need to fix this....
#pellet drops
data3=[]
with open(FoodDrop,'r') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        data3.append(line)
    csvfile.close()
data3=np.array(data3)
Stamps=data3[1:][:,0]
FoodDropList=[]
for i in Stamps:
    FoodDropList.append(float(i)+offset)


data3=[]
with open(b,'r') as csvfile:
    reader=csv.reader(csvfile,delimiter=' ')
    for line in reader:
        data3.append(line)
    csvfile.close()

data3=np.array(data3)
Start=float(data3[0][0])/1000
Foodtime=data3[1:][:,0]
Foodtime2=[]
for x in Foodtime:
    Foodtime2.append(float(x)/1000)
Foodtime=Foodtime2[0::2]
foodnorm=[]
for x in range(len(Foodtime)):
    foodnorm.append((Foodtime[x]-Start)+1.4)
both=np.array(list(zip(FoodDropList,foodnorm)))
diffs=[]
for x in range(len(both)):
    diffs.append(both[x][0]-both[x][1])



########
#Microbehaviors
# data3=[]

# with open(MBFile,'r') as csvfile:
#     reader=csv.reader(csvfile)
#     for line in reader:
#         data3.append(line)
#     csvfile.close()
    
# data3=np.array(data3)
# PMresp=data3[0][1:]
# PMresp2=[]
# for i in PMresp:
#     PMresp2.append(float(i)+offset)
    
# Dispresp=data3[1][1:]
# Dispresp2=[]
# for i in Dispresp:
#     Dispresp2.append(float(i)+offset)

#cue stamps
data3=[]
with open(CueStamps,'r') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        data3.append(line)
    csvfile.close()
data3=np.array(data3)

Stamps=data3[1:][:,0]
CueList=[]
for i in Stamps:
    CueList.append(float(i)+offset)

allnums=list(zip(CueList,diffs))
NormCueList=[]
for num in range(len(allnums)):
    NormCueList.append(allnums[num][0]-allnums[num][1])


#seek actions
data3=[]
with open(SeekStamps,'r') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        data3.append(line)
    csvfile.close()
data3=np.array(data3)
Stamps=data3[1:][:,0]
SeekList=[]
for i in Stamps:
    SeekList.append(float(i)+offset)

allnums=list(zip(SeekList,diffs))
NormSeekList=[]
for num in range(len(allnums)):
    NormSeekList.append(allnums[num][0]-allnums[num][1])

#pellet drops
data3=[]
with open(FoodDrop,'r') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        data3.append(line)
    csvfile.close()
data3=np.array(data3)
Stamps=data3[1:][:,0]
FoodDropList=[]
for i in Stamps:
    FoodDropList.append(float(i)+offset)

allnums=list(zip(FoodDropList,diffs))
NormFDList=[]
for num in range(len(allnums)):
    NormFDList.append(allnums[num][0]-allnums[num][1])

#Pellet Retrievals
data3=[]
with open(FoodRet,'r') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        data3.append(line)
    csvfile.close()
data3=np.array(data3)
Stamps=data3[1:][:,0]
RetrievalList=[]
for i in Stamps:
    RetrievalList.append(float(i)+offset)

allnums=list(zip(RetrievalList,diffs))
NormRetList=[]
for num in range(len(allnums)):
    NormRetList.append(allnums[num][0]-allnums[num][1])


####
#Change list names to work with rest of code

cue=NormCueList
seekresponse=NormSeekList
foodcue=NormFDList
foodresponse=NormRetList
#################################################################################



##################################################
# time decoder


#Relevent Functions
def converttime(time):
    cycle1 = (int(time) >> 12) &  0x1FFF
    cycle2 = (int(time) >> 25) &  0x7f
    seconds = cycle2 + float(cycle1)/8000
    return seconds

def rezero(x):
    x=x-cumetime[0]
    return x

def cam1(x):
    camer1.append(float(x))

def cam1red(x):
    camerred1.append(float(x))

def cam2(x):
    camer2.append(float(x))

def cam2red(x):
    camerred2.append(float(x))

def floater(x):
    x=float(x)
    return x



#########################################
#Start reading in




data=[]

with open(d,'r') as csvfile:
    reader=csv.reader(csvfile,delimiter=' ')
    for line in reader:
        data.append(line)
    csvfile.close()



data2=np.array(data)

#lets just deal with the first 10000 points
#data2=data2[0:60000]

timestamps=data2[:,0]
ints=timestamps.astype(np.int64)

timeconverted=list(map(converttime,ints))


t=1
n=0
time=timeconverted[0]
cumetime=[time]

stop=len(timeconverted)

while t+1 <= stop:

    if (timeconverted[t]-timeconverted[t-1]) > 0:
        time=time + (timeconverted[t]-timeconverted[t-1])
        cumetime.append(time)
        t=t+1


    else:
        n=n+1
        time=timeconverted[t] +(n*128)
        cumetime.append(time)
        t=t+1





cumetime2=list(map(rezero,cumetime))


############################################
#get relevent camera data and rescale the control channel


camer1=[]
camerred1=[]
camer2=[]
camerred2=[]


list(map(cam1,data2[:,1]))
list(map(cam1red,data2[:,2]))
list(map(cam2,data2[:,1]))
list(map(cam2red,data2[:,2]))
####rescale the red channel

y=list(map(float,data2[:,1]))
x=list(map(float,data2[:,2]))
# 
params=np.polyfit(x,y,1)
# 
slope=float(params[0])
intercept=float(params[1])
# 
# 
def rescale(x):
     x = (slope*x)+intercept
     return x
# 
scaledred=list(map(rescale,camerred1))





newcsv=list(zip(cumetime2,camer1,camerred1,scaledred))
newcsv2=list(zip(cumetime2,camer2,camerred2,scaledred))

with open('newfile.csv', 'w') as file:
     wr=csv.writer(file,lineterminator='\n')
     wr.writerow(['time','cam1green','cam1red','scaled_red'])
     for item in newcsv:
         wr.writerow(item)

with open('newfile2.csv', 'w') as file:
     wr=csv.writer(file,lineterminator='\n')
     wr.writerow(['time','cam2green','cam2red','scaled_red(cam1)'])
     for item in newcsv:
         wr.writerow(item)
####################################################

########################################################33
#delta f/f


deltadata=[]
deltadata2=[]


with open('newfile.csv','r') as csvfile:
    reader=csv.reader(csvfile,delimiter=',')
    for line in reader:
        deltadata.append(line)
    csvfile.close()

deltaarray=np.array(deltadata)
stop=len(deltaarray)

#cam2
with open('newfile2.csv','r') as csvfile:
    reader=csv.reader(csvfile,delimiter=',')
    for line in reader:
        deltadata2.append(line)
    csvfile.close()
deltaarray2=np.array(deltadata2)
stop=len(deltaarray2)

cam1deltaf=[]
times2=[]
green=[]
red=[]

#cam2
cam2deltaf=[]
green2=[]
red2=[]

################

## timestamp seconds to sample #


def tosample (x):
    x=round(x,2)
    return x

samplesneeded=list(map(tosample,cue))
seekrespsamplesneeded=list(map(tosample,seekresponse))

foodsamplesneeded=list(map(tosample,foodcue))

foodrespsamplesneeded=list(map(tosample,foodresponse)) ###pellet drop not exit

##
#get all the trial samples for regression
deltascorr=[]
deltareds=[]
deltasgreen=[]
times=[]
dg=[]
dr=[]
samples=[]
t=1
stop=len(samplesneeded)


#####
#get index
t=0
n=1
stop=len(samplesneeded)
indexed=[]

while t+1<= stop:
    ti=float(samplesneeded[t])
     
    if ti <= (float(deltaarray[n][0])): #or ti > (float(deltaarray[n][0])):
        frame=n
        indexed.append(frame) 
        n=1
        t=t+1
    else:
        n=n+1


        
        
##rescale by trial
last=float(len(deltaarray))-2
indexed.append(last)
#indexed.append(195448) #cheeky add to get last trial if needed
    
t=1
stop=len(indexed)


while t+1 <= stop:
    x=indexed[t-1]
    x=int(x)
    diff=indexed[t]
    #one sec prior to cue                      400 and 64
    #gmean=np.mean(list(map(float,deltaarray[(x-400):(x-64)][:,1])))
    #rmean=np.mean(list(map(float,deltaarray[(x-400):(x-64)][:,2])))
    t=t+1
    #41 htz
    x=x-614

    while x < diff:
        deltafgreen=float(deltaarray[x][1])#-gmean)/gmean
        dg.append(deltafgreen)
        samples.append(x)
        times.append(float(deltaarray[x][0]))

        deltafred=float(deltaarray[x][2])#-rmean)/rmean
        dr.append(deltafred)
        x=x+1

    # #get slope and rescale red to green channel;
    params=np.polyfit(dr,dg,1)
    print (params)
    slope=float(params[0])
    intercept=float(params[1])

    def rescale2(x):
        tran = (slope*x)+intercept  # change to "- intercept"
        return tran

    deltas2=list(map(rescale2,dr))

    for x in dg:
        deltasgreen.append(x)
    for x in deltas2:
        deltareds.append(x)

    
    dg=[]
    dr=[]


#corrected deltaf

deltascorr=[]
stop=len(deltareds)

t=0
while t+1 <= stop:
    x=(deltasgreen[t]-deltareds[t])/deltareds[t] ##
   # x=(dg[t]-dr[t])/dr[t] ##    
    deltascorr.append(x)
    t=t+1



#########################
#take data collector


newcsv=list(zip(times,samples,deltasgreen,deltareds,deltascorr))
array=np.array(newcsv)



##################################
#get all the trial samples for regression
seeklist=[]
seek2list=[]
seekcuelist=[]
peaklist=[]
peakforcsv=[]
# peaklist=[]
# zpeaklist=[]
# latpeaklist=[]

#####
#get index
t=0
n=0
stop=len(samplesneeded)#-1
indexed=[]

while t+1<= stop:
    ti=float(samplesneeded[t])
      
    
    if ti <= float(array[n][0]): #or ti > float(array[n][0]):
        frame=n
        indexed.append(frame) 
        n=0
        t=t+1
    else:
        n=n+1
t=0
n=0
stop=len(seekrespsamplesneeded)#-1
srindexed=[]

while t+1<= stop:
    ti=seekrespsamplesneeded[t]
     
    if ti <= float(array[n][0]):# and ti <= float(array[n][0]+.02):
        frame=n
        srindexed.append(frame) 
        n=0
        t=t+1
    else:
        n=n+1
        


######
#action data
t=0
stop=len(srindexed) #srindexed
while t+1<=stop:
    x=srindexed[t] #srindexed
   # print (x)
    y=indexed[t]
    x=int(x)
    y=int(y)
    BL=mean(list((map(float,array[x-103:x-41][:,4]))))
    BLs=np.std(list((map(float,array[x-103:x-41][:,4])))) #[y-4100:y-2][:,3]
    maxchange=float(max(array[x:x+60][:,4]))
    
    ###latency
    #latmaxchange=float(max(array[y:x][:,4]))    
    ##
    seeklatepoch=array[y:x-1][:,4]
    sessmed=np.median(list(map(float,array[1:][:,4])))
    sessstd=sessmed+(robust.mad(list(map(float,array[1:][:,4])))*2)
    sessstdlower=abs(sessmed-(robust.mad(list(map(float,array[1:][:,4])))*2))
    totaltime=array[x-1][0]-array[y][0]
    peaks,_=scipy.signal.find_peaks(seeklatepoch,height=sessstd,distance=41)
    negpeaks,_=scipy.signal.find_peaks(-seeklatepoch,height=sessstdlower,distance=41)
    totalpospeaks=len(peaks)
    totalnegpeaks=len(negpeaks)
    totalpeaks=totalpospeaks+totalnegpeaks
    peakspersec=totalpeaks/totaltime
    
    ampl=0
    negampl=0
    counter=0
    negcounter=0
    for pk in peaks:
        ampl=seeklatepoch[pk]+ampl
        counter=counter+1 
    for pk in negpeaks:
        negampl=seeklatepoch[pk]+negampl
        negcounter=negcounter+1  
    if counter > 0:
        meanAMP=ampl/counter
    else:
        meanAMP='Nan'   
    if negcounter > 0:
        meanNEGAMP=negampl/negcounter
    else:
        meanNEGAMP='Nan'

    peakanalysis=[totalpospeaks,totalnegpeaks,totalpeaks,totaltime,peakspersec,meanAMP,meanNEGAMP]
    peakforcsv.append(peakanalysis)    
    ##################



    peakchange=(maxchange)
    
    #window is 2-sec prior until 3 sec after response
    a=list(map(float,array[x-82:x+123][:,4])) #194
    seeklist.append(list(map(float,a)))
    #peaklist.append(peakchange)
    #latpeaklist.append(latmaxchange)
    t=t+1
    
    #its at 63 htz and there is a .089 sec time lag. So the seek response comes in at sample 132
    hold=[]
    for x in a:
        b=(x-BL)/BLs
        hold.append(b)
    hold=np.array(hold)
    # hold=hold.reshape(-1,2).mean(axis=1)
    # hold=hold.reshape(-1,2).mean(axis=1)
    hold=hold.tolist()
    seek2list.append(hold)
##
t=0
stop=len(indexed) #srindexed
while t+1<=stop:
    x=indexed[t] #srindexed
   # print (x)
    y=indexed[t]
    x=int(x)
    y=int(y)
    BL=mean(list((map(float,array[x-103:x-41][:,4]))))
    BLs=np.std(list((map(float,array[x-103:x-41][:,4])))) #[y-4100:y-2][:,3]
    maxchange=float(max(array[x:x+41][:,4]))
    
    ###latency
    #latmaxchange=float(max(array[y:x][:,4]))    
    ##
    
    peakchange=(maxchange)
    
    #window is 2-sec prior until 3 sec after response
    a=list(map(float,array[x-82:x+123][:,4])) #194
    #seekcuelist.append(list(map(float,a)))
    #peaklist.append(peakchange)
    #latpeaklist.append(latmaxchange)
    t=t+1
    
    #its at 63 htz and there is a .089 sec time lag. So the seek response comes in at sample 132
    hold=[]
    for x in a:
        b=(x-BL)/BLs
        hold.append(b)
    hold=np.array(hold)
    # hold=hold.reshape(-1,2).mean(axis=1)
    # hold=hold.reshape(-1,2).mean(axis=1)
    hold=hold.tolist()
    seekcuelist.append(hold)    
    



###############################################################################################################33

foodlist=[]
food2list=[]
foodcuelist=[]
foodcuelistnonz=[]
Foodpeakcsv=[]
# peaklist=[]
# zpeaklist=[]
# latpeaklist=[]

#####
#get index
t=0
n=0
stop=len(foodsamplesneeded)#-1
indexed=[]

while t+1<= stop:
    ti=float(foodsamplesneeded[t])
      
    
    if ti <= float(array[n][0]): #or ti > float(array[n][0]):
        frame=n
        indexed.append(frame) 
        n=0
        t=t+1
    else:
        n=n+1
t=0
n=0
stop=len(foodrespsamplesneeded)#-1
frindexed=[]

while t+1<= stop:
    ti=foodrespsamplesneeded[t]
     
    if ti <= float(array[n][0]):# and ti <= float(array[n][0]+.02):
        frame=n
        frindexed.append(frame) 
        n=0
        t=t+1
    else:
        n=n+1
        


######
#action data
t=0
stop=len(frindexed) #srindexed
while t+1<=stop:
    x=frindexed[t] #srindexed
   # print (x)
    y=indexed[t]
    x=int(x)
    y=int(y)
    BL=mean(list((map(float,array[x-103:x-41][:,4]))))
    BLs=np.std(list((map(float,array[x-103:x-41][:,4])))) #[y-4100:y-2][:,3]
    maxchange=float(max(array[x:x+60][:,4]))
    
    ###latency
    #peakanalysisfood
    itiepoch=array[x:x+585][:,4]
    sessmed=np.median(list(map(float,array[1:][:,4])))
    sessstd=sessmed+(robust.mad(list(map(float,array[1:][:,4])))*2)
    sessstdlower=abs(sessmed-(robust.mad(list(map(float,array[1:][:,4])))*2))
    Foodtotaltime=14.28
    peaks,_=scipy.signal.find_peaks(itiepoch,height=sessstd,distance=41)
    negpeaks,_=scipy.signal.find_peaks(-itiepoch,height=sessstdlower,distance=41)
    foodtotalpospeaks=len(peaks)
    foodtotalnegpeaks=len(negpeaks)
    foodtotalpeaks=foodtotalpospeaks+foodtotalnegpeaks
    foodpeakspersec=foodtotalpeaks/Foodtotaltime
    
    ampl=0
    negampl=0
    counter=0
    negcounter=0
    for pk in peaks:
        ampl=itiepoch[pk]+ampl
        counter=counter+1 
    for pk in negpeaks:
        negampl=itiepoch[pk]+negampl
        negcounter=negcounter+1  
    if counter > 0:
        FoodmeanAMP=ampl/counter
    else:
        FoodmeanAMP='Nan'   
    if negcounter > 0:
        FoodmeanNEGAMP=negampl/negcounter
    else:
        FoodmeanNEGAMP='Nan'

    Foodpeakanalysis=[foodtotalpospeaks,foodtotalnegpeaks,foodtotalpeaks,Foodtotaltime,foodpeakspersec,FoodmeanAMP,FoodmeanNEGAMP]
    Foodpeakcsv.append(Foodpeakanalysis)   
    #latmaxchange=float(max(array[y:x][:,4]))    
    ##
    
    peakchange=(maxchange)
    
    #window is 2-sec prior until 3 sec after response
    a=list(map(float,array[x-82:x+582][:,4])) #194
    foodlist.append(list(map(float,a)))
    #peaklist.append(peakchange)
    #latpeaklist.append(latmaxchange)
    t=t+1
    
    #its at 63 htz and there is a .089 sec time lag. So the seek response comes in at sample 132
    hold=[]
    for x in a:
        b=(x-BL)/BLs
        hold.append(b)
    hold=np.array(hold)
    # hold=hold.reshape(-1,2).mean(axis=1)
    # hold=hold.reshape(-1,2).mean(axis=1)
    hold=hold.tolist()
    food2list.append(hold)
##
t=0
stop=len(indexed) 
while t+1<=stop:
    x=indexed[t] 
   # print (x)
    y=indexed[t]
    x=int(x)
    y=int(y)
    BL=mean(list((map(float,array[x-41:x-1][:,4]))))
    BLs=np.std(list((map(float,array[x-41:x-1][:,4])))) #[y-4100:y-2][:,3]
    maxchange=float(max(array[x:x+60][:,4]))
    
    ###latency
    #latmaxchange=float(max(array[y:x][:,4]))    
    ##
    
    peakchange=(maxchange)
    
    #window is 2-sec prior until 3 sec after response
    a=list(map(float,array[x-82:x+123][:,4])) #194
    foodcuelistnonz.append(list(map(float,a)))
    #peaklist.append(peakchange)
    #latpeaklist.append(latmaxchange)
    t=t+1
    
    #its at 63 htz and there is a .089 sec time lag. So the seek response comes in at sample 132
    hold=[]
    for x in a:
        b=(x-BL)/BLs
        hold.append(b)
    hold=np.array(hold)
    # hold=hold.reshape(-1,2).mean(axis=1)
    # hold=hold.reshape(-1,2).mean(axis=1)
    hold=hold.tolist()
    foodcuelist.append(hold) 







####
#write it out
import os


name=CueStamps.split('_')
name=name[0]+name[1]+'mPFC_Analysis'


os.mkdir(name)
os.chdir(name)



with open('seekres.csv', 'w') as file:
    wr=csv.writer(file,lineterminator='\n')
    for item in seeklist:
        wr.writerow(item)
        
with open('seekreszscore.csv', 'w') as file:
    wr=csv.writer(file,lineterminator='\n')
    for item in seek2list:
        wr.writerow(item)

with open('seeklatencyPeaks.csv', 'w') as file:
    wr=csv.writer(file,lineterminator='\n')
    wr.writerow(['Pos Peaks', 'Neg Peaks','Total Peaks', 'Latency', 'Peaks/sec','meanAMP','meanNEGAMP'])    
    for item in peakforcsv:
        wr.writerow(item)

with open('seekcuezscore.csv', 'w') as file:
    wr=csv.writer(file,lineterminator='\n')
    for item in seekcuelist:
        wr.writerow(item)

with open('foodcuezscore.csv', 'w') as file:
    wr=csv.writer(file,lineterminator='\n')
    for item in foodcuelist:
        wr.writerow(item)
with open('foodITIz.csv', 'w') as file:
    wr=csv.writer(file,lineterminator='\n')
    for item in food2list:
        wr.writerow(item)

with open('ITIPeaks.csv', 'w') as file:
    wr=csv.writer(file,lineterminator='\n')
    wr.writerow(['Pos Peaks', 'Neg Peaks','Total Peaks', 'Latency', 'Peaks/sec','meanAMP','meanNEGAMP'])    
    for item in Foodpeakcsv:
        wr.writerow(item) 



####
#IDK maybe use PANDAS
import pandas

df=pandas.read_csv('seekres.csv')

tpose=df.transpose()
tpose.to_csv('seekres.csv')

df=pandas.read_csv('seekreszscore.csv')

tpose=df.transpose()
tpose.to_csv('seekreszscore.csv')

df=pandas.read_csv('seekcuezscore.csv')

tpose=df.transpose()
tpose.to_csv('seekcuezscore.csv')


df=pandas.read_csv('foodITIz.csv')

tpose=df.transpose()
tpose.to_csv('foodITIz.csv')

df=pandas.read_csv('foodcuezscore.csv')

tpose=df.transpose()
tpose.to_csv('foodcuezscore.csv')

os.chdir('../')



#cam2
name=CueStamps.split('_')
name=name[0]+name[1]+'VTA_Analysis'


os.mkdir(name)
os.chdir(name)



with open('seekres.csv', 'w') as file:
    wr=csv.writer(file,lineterminator='\n')
    for item in seeklist:
        wr.writerow(item)
        
with open('seekreszscore.csv', 'w') as file:
    wr=csv.writer(file,lineterminator='\n')
    for item in seek2list:
        wr.writerow(item)

with open('seeklatencyPeaks.csv', 'w') as file:
    wr=csv.writer(file,lineterminator='\n')
    wr.writerow(['Pos Peaks', 'Neg Peaks','Total Peaks', 'Latency', 'Peaks/sec','meanAMP','meanNEGAMP'])    
    for item in peakforcsv:
        wr.writerow(item)

with open('seekcuezscore.csv', 'w') as file:
    wr=csv.writer(file,lineterminator='\n')
    for item in seekcuelist:
        wr.writerow(item)

with open('foodcuezscore.csv', 'w') as file:
    wr=csv.writer(file,lineterminator='\n')
    for item in foodcuelist:
        wr.writerow(item)
with open('foodITIz.csv', 'w') as file:
    wr=csv.writer(file,lineterminator='\n')
    for item in food2list:
        wr.writerow(item)

with open('ITIPeaks.csv', 'w') as file:
    wr=csv.writer(file,lineterminator='\n')
    wr.writerow(['Pos Peaks', 'Neg Peaks','Total Peaks', 'Latency', 'Peaks/sec','meanAMP','meanNEGAMP'])    
    for item in Foodpeakcsv:
        wr.writerow(item) 



####
#IDK maybe use PANDAS
import pandas

df=pandas.read_csv('seekres.csv')

tpose=df.transpose()
tpose.to_csv('seekres.csv')

df=pandas.read_csv('seekreszscore.csv')

tpose=df.transpose()
tpose.to_csv('seekreszscore.csv')

df=pandas.read_csv('seekcuezscore.csv')

tpose=df.transpose()
tpose.to_csv('seekcuezscore.csv')


df=pandas.read_csv('foodITIz.csv')

tpose=df.transpose()
tpose.to_csv('foodITIz.csv')

df=pandas.read_csv('foodcuezscore.csv')

tpose=df.transpose()
tpose.to_csv('foodcuezscore.csv')

os.chdir('../')