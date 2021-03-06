

class Bayes:
    def __init__(self, hypothesis_list, prior_list, observations_list, likelihood_array):
        self.hypothesis_list_ = hypothesis_list
        self.prior_list_ = prior_list
        self.likelihood_array_ = likelihood_array
        self.observations_list_ = observations_list

    """
        function likelihood takes:d
            - observation
            - hypothesis

        Find the index of the observation and the hypothesis
        These are necessary to find the relevant value in the likelihood array
    """

    def likelihood(self, observation, hypothesis):

        pos_hyp = self.hypothesis_list_.index(hypothesis)
        pos_obs = self.observations_list_.index(observation)
        lk = self.likelihood_array_[pos_hyp][pos_obs]
        return lk

    """
        function norm_constant takes
            - observation

        For each hypothesis calculate p(O | H) * p(H) and add them together
    """

    def norm_constant(self, observation):
        norm_cst = 0
        for index, item in enumerate(self.hypothesis_list_):
            prior = self.prior_list_[index]
            norm_cst = norm_cst + prior*self.likelihood(observation, item)
        return norm_cst

    """
        function single_posterior_update takes
            - observation
            - list of priors

        Using Bayes rule calculate p(H | O)
    """

    def single_posterior_update(self, observation, priors_list):
        posterior_list = []

        for count, prior in enumerate(priors_list):
            likelihood = self.likelihood(
                observation, self.hypothesis_list_[count])
            const = self.norm_constant(observation)
            posterior = (prior*likelihood)/const
            posterior_list.append(posterior)

        return posterior_list

    """
        function compute_posterior takes
                - list of observation

        Calculates P(H | O_1, O_2, ..., O_n)
    """

    def compute_posterior(self, observations_list):
        posterior = 0
        for observation in observations_list:
            posterior = self.single_posterior_update(
                observation, self.prior_list_)
            self.prior_list_ = posterior

        return posterior


if __name__ == "__main__":

    #Problem 1
    print("Problem 1")
    c_hypos = ["Bowl1", "Bowl2"]
    c_priors = [0.5, 0.5]
    c_obs_list = ["chocolate", "vanilla"]
    c_likelihood = [[15/50, 35/50], [30/50, 20/50]]
    cookie_b = Bayes(c_hypos, c_priors, c_obs_list, c_likelihood)

    l = cookie_b.likelihood("chocolate", "Bowl1")
    print("likelihood(chococolate, Bowl1) =  ", l)

    q1 = cookie_b.single_posterior_update("vanilla", [0.5, 0.5]) # Probabilities of having picked bowl 1 and bowl 2 given we observe a vanilla cookie.

    #print("single_posterior_update(vanilla, [0.5, 0.5]) =  ", q1)
    print("Probability of having picked bowl 1 after seeing a vanilla cookie is {:05.3f} ".format(q1[c_hypos.index("Bowl1")]))

    q2 = cookie_b.compute_posterior(["chocolate", "vanilla"]) # Probabilities of having picked bowl 1 and bowl 2 given we observe a cholcolate and a vanilla cookie.

    #print("chocolate, vanilla =  ", q2)
    print("Probability of having picked bowl 2 after seeing  a chocolate and a vanilla cookie is {:05.3f}".format(q2[c_hypos.index("Bowl2")]))


    #Problem2
    print("Problem 2")
    a_hypos = ["Beginner", "Intermediate", "Advanced", "Expert"]
    a_priors = [0.25, 0.25, 0.25, 0.25]
    a_obs_list = ["yellow", "red", "blue", "black", "white"]
    a_likelihood = [[0.05, 0.1, 0.4, 0.25, 0.2], [0.1, 0.2, 0.4, 0.2, 0.1], [
        0.2, 0.4, 0.25, 0.1, 0.05], [0.3, 0.5, 0.125, 0.05, 0.025]]
    archer_b = Bayes(a_hypos, a_priors, a_obs_list, a_likelihood)

    a_obs = ["yellow", "white", "blue", "red", "red", "blue"]
    q3 = archer_b.compute_posterior(a_obs)

    print("Probability that the archer is intermediate given we see yellow, white, blue, red, red, blue = {:05.3f} ".format(q3[a_hypos.index("Intermediate")]))

    index_max = q3.index(max(q3))
    most_prob_level = a_hypos[index_max]
    print("Most probable level of the archer is: %s" % most_prob_level)
    ## Code for printing to file
    result = "{:.3f}\n{:.3f}\n{:.3f}\n{}".format(q1[c_hypos.index("Bowl1")], q2[c_hypos.index("Bowl2")],
                                                 q3[a_hypos.index("Intermediate")], most_prob_level)
    f = open("group_23.txt", "w")
    f.write(result)
    f.close()
