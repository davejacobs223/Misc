#! bin/usr/env python 

#A Fiber Photometry Script that is a work in progress

import scipy
import csv
import numpy as np
from numpy import mean
from pylab import *
from scipy import signal
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns

######################################################################################33




#get timestamp data and load files
a='afil.csv'#cue file for GT
b='afil.csv'
d='afile.csv'#data file of camera
Stamps='anotherfile.csv'



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





#cue stamps
data3=[]
with open(Stamps,'r') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        data3.append(line)
    csvfile.close()
data3=np.array(data3)

Stamps=data3[1:][:,0]
CueList=[]
for i in Stamps:
    CueList.append(float(i)+offset)





####
#Change list names to work with rest of code

cue=CueList

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
####rescale the red channel for the whole trace (not used)

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


with open('newfile.csv', 'w') as file:
     wr=csv.writer(file,lineterminator='\n')
     wr.writerow(['time','cam1green','cam1red','scaled_red'])
     for item in newcsv:
         wr.writerow(item)

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


cam1deltaf=[]
times2=[]
green=[]
red=[]


################

## timestamp seconds to sample #


def tosample (x):
    x=round(x,2)
    return x

samplesneeded=list(map(tosample,cue))


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
     
    if ti <= (float(deltaarray[n][0])):
        frame=n
        indexed.append(frame) 
        n=1
        t=t+1
    else:
        n=n+1


        
        
##rescale by trial
last=float(len(deltaarray))-2
indexed.append(last)

    
t=1
stop=len(indexed)


while t+1 <= stop:
    x=indexed[t-1]
    x=int(x)
    diff=indexed[t]
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





####
#write it out
import os


name=CueStamps.split('_')
name=name[0]+name[1]+'Analysis'


os.mkdir(name)
os.chdir(name)



with open('seekres.csv', 'w') as file:
    wr=csv.writer(file,lineterminator='\n')
    for item in seeklist:
        wr.writerow(item)
        




####
#IDK maybe use PANDAS
import pandas

df=pandas.read_csv('seekres.csv')



os.chdir('../')



#! bin/usr/env python 
import pandas as pd
import numpy as np
import csv


a=pd.read_csv('dataa.csv',header=None)

headers=a.columns[1:]

masterlist=[]

def getAUC(column,df=a):
    subject=df[column][3]
    session=df[column][2]
    block=df[column][0]
    shock=df[column][1]


    preaction=np.trapz(df[column][44:86].to_numpy(dtype=float))
    postaction=np.trapz(df[column][86:128].to_numpy(dtype=float))
    changescore=postaction-preaction
    food=np.trapz(df[column][128:].to_numpy(dtype=float))
    return (block,shock,session,subject,preaction,postaction,changescore,food)


for name in headers:
    masterlist.append(getAUC(name))

AUCdf = pd.DataFrame.from_records(masterlist, columns =['Block','Shock', 'Session','Subject', 'preaction','postaction','actiondelta','food'])

AUCdf.sort_values(by=['Shock','Session','Block','Subject']).transpose().to_csv(region+'seek_AUC.csv',header=None)