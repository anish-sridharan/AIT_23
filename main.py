from Bayes import Bayes

if __name__ == "__main__" :
    #Problem 1
    print("Answer to Problem1")
    hypos = ["Bowl1", "Bowl2"]
    priors = [0.5,0.5]
    obs_list = ["chocolate", "vanilla"]
    likelihood = [[15/50 , 35/50], [30/50, 20/50]]
    b = Bayes(hypos,priors,obs_list,likelihood)

    l = b.likelihood("chocolate", "Bowl1")
    round(l,3)
    print("likelihood(chococolate, Bowl1) =  " , l)

    l = b.single_posterior_update("vanilla", [0.5, 0.5])
   
    print("single_posterior_updatedate(vanilla, [0.5, 0.5]) =  " , l)

    l =b.compute_posterior(["chocolate", "vanilla"])
   
    print("chocolate, vannila =  " , l)


    #Problem2
    print("Answer to Problem 2")
    hypos = ["Beginner", "Intermediate","Advanced","Expert"]
    priors = [0.25,0.25,0.25,0.25]
    obs_list = ["yellow", "red","blue","black","white"]
    obs = ["yellow", "white","blue","red","red","blue"]
    likelihood = [[0.05 , 0.1, 0.4, 0.25, 0.2], [0.1, 0.2, 0.4, 0.2, 0.1], [0.2, 0.4, 0.25, 0.1, 0.05],[0.3, 0.5, 0.125, 0.05, 0.025]]
    b = Bayes(hypos,priors,obs_list,likelihood)

   

    l =b.compute_posterior(obs)
   
    print("yellow, white, blue, red, red, blue =  " , l)