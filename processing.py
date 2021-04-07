import numpy as np
from PIL import Image as im 

#resolusion of output images
img_width = 12
img_height = 50

#choose txt file
mode = 'run'

#loop - append accelerometr data to list
for k in range(0,24):
    i=0
    j=0
    xPom = []
    yPom = []
    zPom = []
    pom = []

    with open('data/' + mode + '.txt','r') as file: 
    
        # reading each line  
        for line in file: 
            #skip comment lines
            if "#" not in line:  
                i+=1
                #choosing of the scope of data
                if(i>k*200+201):
                    break
                if(i>k*200+1):
                    j=0
                    for word in line.split(): 
                        j+=1
                        if(j==1):
                            xPom.append(int(abs(float(word))*3))
                        elif(j==2):
                            yPom.append(int(abs(float(word))*3))
                        elif(j==3):
                            zPom.append(int(abs(float(word))*3))


    #checking if size of lists are fine
    if(len(xPom) == len(yPom) == len(zPom)):

        for x in xPom:
            pom.append(x)
        
        for y in yPom:
            pom.append(y)

        for z in zPom:
            pom.append(z)

        #convert lists to numpy array
        nppom = np.array(pom)

        #reshape numpy array
        reshaped_data = np.reshape(nppom, (img_height, img_width)) 

        #convert data to uint8 
        uint_data = np.uint8(reshaped_data)

        #control check of uint_data
        for x in uint_data:
            print(x)
    
        img = im.fromarray(uint_data)
        img.save('output/' + mode + str(k) + '.png')

    else:
        raise Exception ('Wrong sizes of lists xPom, yPom, zPom.')
        break

file.close()


    

                
                    
                
                                         


