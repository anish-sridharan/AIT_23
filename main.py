from Bayes import Bayes
import numpy as np
if __name__ == "__main__" :
    file1 = open("Group_23.txt","w")
    #Problem 1
    print("Answer to Problem1")
    hypos = ["Bowl1", "Bowl2"]
    priors = [0.5,0.5]
    obs_list = ["chocolate", "vanilla"]
    likelihood = [[15/50 , 35/50], [30/50, 20/50]]
    #likelihood 0.3
    b = Bayes(hypos,priors,obs_list,likelihood)



    Answer1 = b.single_posterior_update("vanilla", [0.5, 0.5])
   
    print("single_posterior_updatedate(vanilla, [0.5, 0.5]) =  " , Answer1)
    file1.write(str(Answer1[0])+"\n")
    #0.636

    Answer2 =b.compute_posterior(["chocolate", "vanilla"])
   
    print("chocolate, vannila =  " , Answer2)
    file1.write(str(Answer2[1])+"\n")
    #0.534


    #Problem2
    print("Answer to Problem 2")
    hypos = ["Beginner", "Intermediate","Advanced","Expert"]
    priors = [0.25,0.25,0.25,0.25]
    obs_list = ["yellow", "red","blue","black","white"]
    obs = ["yellow", "white","blue","red","red","blue"]
    likelihood = [[0.05 , 0.1, 0.4, 0.25, 0.2], [0.1, 0.2, 0.4, 0.2, 0.1], [0.2, 0.4, 0.25, 0.1, 0.05],[0.3, 0.5, 0.125, 0.05, 0.025]]
    b = Bayes(hypos,priors,obs_list,likelihood)

   

    Answer3=b.compute_posterior(obs)
   
    print("yellow, white, blue, red, red, blue =  " , Answer3)
    file1.write(str(Answer3[1])+"\n")
    Answer4=hypos[np.argmax(np.array(Answer3))]
    file1.write(Answer4)
    #Intermidiate 0.306
    #Advanced