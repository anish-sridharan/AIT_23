

class Bayes:
    def __init__(self, hypothesis_list, prior_list, observations_list, likelihood_array):
        self.hypothesis_list_ = hypothesis_list
        self.prior_list_ = prior_list
        self.likelihood_array_ = likelihood_array
        self.observations_list_ = observations_list

    """
        function likelihood takes:
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
            posterior = round(posterior, 3)
            posterior_list.append(posterior)

        return posterior_list

    """
        function compute_posterior takes
                - list of observation

        Calculates P(H | O_1, O_2, ..., O_n) = P(O_1 | H) * P(O_2 | H) * P(H) / P(O_1, O_2)
    """

    def compute_posterior(self, observations_list):
        for observation in observations_list:
            posterior = self.single_posterior_update(
                observation, self.prior_list_)
            self.prior_list_ = posterior

        return posterior


if __name__ == "__main__":

    #Problem 1
    print("Answer to Problem1")
    hypos = ["Bowl1", "Bowl2"]
    priors = [0.5, 0.5]
    obs_list = ["chocolate", "vanilla"]
    likelihood = [[15/50, 35/50], [30/50, 20/50]]
    b = Bayes(hypos, priors, obs_list, likelihood)

    l = b.likelihood("chocolate", "Bowl1")
    round(l, 3)
    print("likelihood(chococolate, Bowl1) =  ", l)

    l = b.single_posterior_update("vanilla", [0.5, 0.5])

    print("single_posterior_updatedate(vanilla, [0.5, 0.5]) =  ", l)

    l = b.compute_posterior(["chocolate", "vanilla"])

    print("chocolate, vannila =  ", l)

    #Problem2
    print("Answer to Problem 2")
    hypos = ["Beginner", "Intermediate", "Advanced", "Expert"]
    priors = [0.25, 0.25, 0.25, 0.25]
    obs_list = ["yellow", "red", "blue", "black", "white"]
    obs = ["yellow", "white", "blue", "red", "red", "blue"]
    likelihood = [[0.05, 0.1, 0.4, 0.25, 0.2], [0.1, 0.2, 0.4, 0.2, 0.1], [
        0.2, 0.4, 0.25, 0.1, 0.05], [0.3, 0.5, 0.125, 0.05, 0.025]]
    b = Bayes(hypos, priors, obs_list, likelihood)

    l = b.compute_posterior(obs)

    print("yellow, white, blue, red, red, blue =  ", l)

