from Bayes import Bayes

if __name__ == "__main__" :
    hypos = ["Bowl1", "Bowl2"]
    priors = [0.5,0.5]
    obs = ["chocolate", "vanilla"]
    likelihood = [[15/50 , 35/50], [30/50, 20/50]]
    b = Bayes(hypos,priors,obs,likelihood)

    l = b.likelihood("chocolate", "Bowl1")
    round(l,)
    print("likelihood(chococolate, Bowl1) =  " , l)