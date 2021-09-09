class Bayes:
    def __init__(self, hypos, priors, obs, like):
        self.hypos = hypos
        self.priors = priors
        self.obs = obs
        self.like = like

    def likelihood(self, true_obs, true_hypo):
        hypo_pos = self.hypos.index(true_hypo)
        obs_pos = self.obs.index(true_obs)

        return self.like[hypo_pos][obs_pos]

    def norm_constant(self, true_obs):
        obs_pos = self.obs.index(true_obs)
        norm = 0
        for x in range(len(self.like)):
            norm = norm + self.like[x][obs_pos]

        return norm/len(self.like)

    def single_posterior_update(self, true_obs, prior):
        posts = [None] * len(self.hypos)
        norm = Bayes.norm_constant(self, true_obs)
        obs_pos = self.obs.index(true_obs)

        for x in range(len(self.hypos)):
            posts[x] = (prior[x]*self.like[x][obs_pos])/norm
        
        return posts
    
    def compute_posterior(self, observations):
        compound_like = [1] * len(self.hypos)
        compound_norm = 1
        posts = [None] * len(self.hypos)
        first_go = True

        for x in observations:
            if first_go:
                posts =  Bayes.single_posterior_update(self, x, self.priors)
                first_go = False
            else:
                obs_pos = self.obs.index(x)
                for i in range(len(self.hypos)):
                    posts[i] = posts[i] * (self.like[i][obs_pos]/Bayes.norm_constant(self,x))       
        
        return posts


if __name__ == "__main__":
    hypos = ["Bowl1", "Bowl2"]
    priors = [0.5, 0.5]
    obs = ["chocolate", "vanilla"]
    # e.g. likelihood[0][1] corresponds to the likehood of Bowl1 and vanilla, or 35/50
    likelihood = [[15/50, 35/50], [30/50, 20/50]]
    
    b = Bayes(hypos, priors, obs, likelihood)
    
    l = b.likelihood("chocolate", "Bowl1")
    print("likelihood(chocolate, Bowl1) = %s " % l)
    
    n_c = b.norm_constant("vanilla")
    print("normalizing constant for vanilla: %s" % n_c)
    
    p_1 = b.single_posterior_update("vanilla", [0.5, 0.5])
    print("vanilla - posterior: %s" % p_1)

    # Question 1: 0.636

    p_2 = b.compute_posterior(["chocolate", "vanilla"])
    print("chocolate, vanilla - posterior: %s" % p_2)

    # Question 2: 0.485

    a_hypos = ["beginner", "intermediate", "advanced", "expert"]
    a_priors = [0.25, 0.25, 0.25, 0.25]
    a_obs = ["yellow", "red", "blue", "black", "white"]
    a_likelihood = [[0.05, 0.1, 0.4, 0.25, 0.2], [0.1, 0.2, 0.4, 0.2, 0.1], [0.2, 0.4, 0.25, 0.1, 0.05], [0.3, 0.5, 0.125, 0.05, 0.025]]

    a = Bayes(a_hypos, a_priors, a_obs, a_likelihood)

    p_a = a.compute_posterior(["yellow", "white", "blue", "red", "red", "blue"])
    print("Archer posteriors: %s" % p_a)

    # Question 3: 0.135
    # Question 4: advanced
