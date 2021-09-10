import numpy


class Bayes:

    def __init__(self, hypothesis, priors, likelihoods) -> None:
        super().__init__()

        self.hypothesis = hypothesis
        self.priors = priors
        self.likelihoods = likelihoods

    def likelihood(self, observation, hypothesis):
        return self.likelihoods[hypothesis][observation]

    def norm_constant(self, observation, priors):
        res = 0
        for h in self.hypothesis:
            res += (priors[h] * self.likelihood(observation, h))
        return res

    def single_posterior_update(self, observation, priors):
        c = self.norm_constant(observation, priors)
        res = {}
        for hypothesis, prior in priors.items():
            res[hypothesis] = prior * self.likelihood(observation, hypothesis) / c
        return res
    
    def compute_posterior(self, observations):
        posteriors = self.priors
        for observation in observations:
            posteriors = self.single_posterior_update(observation, priors=posteriors)
        return posteriors

if __name__ == "__main__":

    """
        COOKIES
    """
    def likelihoods(vanilla, chocolate):
        t = vanilla + chocolate
        return { 'vanilla': vanilla / t, 'chocolate': chocolate / t }

    bayes = Bayes(hypothesis=['b1', 'b2'], priors={'b1': 0.5, 'b2': 0.5}, likelihoods={
        'b1': likelihoods(vanilla=35, chocolate=15),
        'b2': likelihoods(vanilla=20, chocolate=30)
    })

    print("likelihood(chocolate, Bowl1)", bayes.likelihood("chocolate", "b1"))
    print("Cookie vanilla posterior", bayes.single_posterior_update("vanilla", priors={'b1': 0.5, 'b2': 0.5})) # Q1 = 0.636
    print("Cookie vanilla & chocolate posterior", bayes.compute_posterior(["chocolate", "vanilla"])) # Q2: 0.485

    """
        ARCHERS
    """

    archer_likelihood = {
        'beginner': { 'yellow': 0.05, 'red': 0.1, 'blue': 0.4, 'black': 0.25, 'white': 0.2 },
        'intermediate': { 'yellow': 0.1, 'red': 0.2, 'blue': 0.4, 'black': 0.2, 'white': 0.1 },
        'advanced': { 'yellow': 0.2, 'red': 0.4, 'blue': 0.25, 'black': 0.1, 'white': 0.05 },
        'expert': { 'yellow': 0.3, 'red': 0.5, 'blue': 0.125, 'black': 0.05, 'white': 0.025 },
    }

    bayes = Bayes(hypothesis=["beginner", "intermediate", "advanced", "expert"],
                priors={"beginner": 0.25, "intermediate": 0.25, "advanced": 0.25, "expert": 0.25},
                likelihoods=archer_likelihood)

    stats = bayes.compute_posterior(["yellow", "white", "blue", "red", "red", "blue"])
    print("Archer posteriors ", stats) # Q3 'intermediate': 0.135
    print('Most likely rank', max(stats, key=stats.get)
)