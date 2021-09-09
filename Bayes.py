class Bayes:
    def __init__(self, hypothesis, prior_hyp, possible_obs, likelihood):
        self.hypos = hypothesis
        self.priors = prior_hyp
        self.obs = possible_obs
        self.ll = likelihood

    def likelihood(self, ob, hypothesis):
        # Find the index of the observation and the hypothesis
        # These are necessary to find the relevant value in the likelihood array
        index_ob = self.obs.index(ob)
        index_hypo = self.hypos.index(hypothesis)
        return self.ll[index_hypo][index_ob]

    def norm_constant(self, ob):
        normalizing_constant = 0
        # For each hypothesis calculate p(O | H) * p(H) and add them together
        for hypothesis in self.hypos:
            index_hypo = self.hypos.index(hypothesis)
            normalizing_constant += self.priors[index_hypo] * self.likelihood(ob, hypothesis)
        return normalizing_constant

    def single_posterior_update(self, ob, priors):
        # Using Bayes rule calculate p(H | O)
        posterior = []
        for index, prior in enumerate(priors):
            temp_post = prior * self.likelihood(ob, self.hypos[index]) / self.norm_constant(ob)
            posterior.append(temp_post)
        return posterior

    def compute_posterior(self, obs):
        # Calculate P(H | O_1, O_2, ..., O_n) = P(O_1 | H) * P(O_2 | H) * P(H) / P(O_1, O_2)
        posterior = []
        for index_h, hypo in enumerate(self.hypos):
            temp_post = 0
            for i, ob in enumerate(obs):
                if i == 0:
                    temp_post = self.single_posterior_update(ob, self.priors)[index_h]
                else:
                    temp_post *= self.likelihood(ob, hypo) / self.norm_constant(ob)
            posterior.append(temp_post)
        return posterior



if __name__ == '__main__':
    print("Cookie Problem")
    c_hypos = ["Bowl1", "Bowl2"]
    c_priors = [0.5, 0.5]
    c_obs = ["chocolate", "vanilla"]
    c_likelihood = [[15/50, 35/50],[30/50, 20/50]]
    b = Bayes(c_hypos, c_priors, c_obs, c_likelihood)

    l = b.likelihood("chocolate", "Bowl1")
    print("likelihood(chocolate, Bowl1) = %s" % l)


    n_c = b.norm_constant("vanilla")
    print("normalizing constant for vanilla: %s" % n_c)

    p_1 = b.single_posterior_update("vanilla", [0.5, 0.5])
    print("vanilla - posterior: %s" % p_1)

    p_2 = b.compute_posterior(["chocolate", "vanilla"])
    print("chocolate, vanilla - posterior %s" % p_2)

    print("=======================================================")
    print("Archer")
    a_hypos = ["Beginner", "Intermediate", "Advanced", "Expert"]
    a_priors = [0.25 for i in a_hypos]
    a_obs = ["Yellow", "Red", "Blue", "Black", "White"]
    a_likelihood = [[0.05, 0.1, 0.4, 0.25, 0.2], [0.1, 0.2, 0.4, 0.2, 0.1], [0.2, 0.4, 0.25, 0.1, 0.05], [0.3, 0.5, 0.125, 0.05, 0.025]]

    arch_b = Bayes(a_hypos, a_priors, a_obs, a_likelihood)
    posterior_levels = arch_b.compute_posterior(["Yellow", "White", "Blue", "Red", "Red", "Blue"])
    print("Posterior: %s" % posterior_levels)

    index_max = posterior_levels.index(max(posterior_levels))
    most_prob_level = a_hypos[index_max]
    print("Most probable level: %s" % most_prob_level)

    ## Code for printing to file
    result = "{:.3f}\n{:.3f}\n{:.3f}\n{}".format(p_1[c_hypos.index("Bowl1")], p_2[c_hypos.index("Bowl2")],
                                                 posterior_levels[a_hypos.index("Intermediate")], most_prob_level)
    f = open("group_23.txt", "w")
    f.write(result)
    f.close()
